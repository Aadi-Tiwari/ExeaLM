"""Post-training step 3: reinforcement learning with GRPO (Group Relative Policy Optimization, from DeepSeekMath and
used for DeepSeek-R1).

For each prompt, sample a GROUP of answers, score each with a reward (for example: is the final answer right?), and
push up the answers that scored above their group's average. No separate value model is needed, which is what makes
GRPO cheap enough for us.

    advantage_i = (reward_i - mean(group rewards)) / std(group rewards)
    loss = - mean over tokens of min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + beta * KL(policy || reference)

Training (after SFT or DPO, with the tokenizer the checkpoint was trained with):
    python -m exealm.train.grpo --ckpt runs/sft/model.pt --tokenizer tokenizer --data tasks.jsonl.gz --steps 500

Data: one task per line of a .jsonl or .jsonl.gz file (exealm/data/post.py writes them):
    {"prompt": "...", "answer": "...", "reward": "math"}      or "messages": [...] in place of "prompt"
`reward` names a function in REWARDS (default "math"):
    math    the last \\boxed{...} of the answer, or else its last number, equals `answer`
    exact   the answer, stripped, equals `answer`
    code    `answer` holds test code; the first Python block of the answer runs with it and must exit cleanly.
            This RUNS MODEL-WRITTEN CODE. It is refused unless you pass --allow_code_exec, and you should only
            pass that inside a container or a throwaway VM.

One step: pick --prompts tasks, sample --group answers for each (one generate call per task, so every row of a call
has the same prompt and no padding), score them, then take --epochs optimizer steps on those rows. A frozen copy of
the starting model is the reference for the KL term. An answer that reaches <|im_end|> is scored with that token, so
the model also learns when to stop.

  TODO(post-training): rewards for format and Japanese tasks; instruction-following constraints (RLVR).
  TODO(post-training): multi-agent rewards: a swarm of agents shares a board and is rewarded together for solving a
                       task none could solve alone. This is how the AG circuits learn to talk (docs/swarms.md).
"""
import argparse
import copy
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import re

import numpy as np
import torch


def group_advantages(rewards, group_size, eps=1e-6):
    """rewards (N,) with N = prompts * group_size, grouped consecutively."""
    r = rewards.float().view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_loss(logp, logp_old, logp_ref, adv, mask, clip=0.2, beta=0.04):
    """logp, logp_old, logp_ref: (N, L) log-probs of the sampled completion tokens under the current, sampling-time and
    reference (frozen SFT) models; adv (N,); mask (N, L) 1 on completion tokens."""
    ratio = torch.exp(logp - logp_old)
    a = adv.view(-1, 1)
    pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    kl = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1            # unbiased, always >= 0
    per_tok = pg + beta * kl
    mask = mask.float()
    return ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def token_logprobs(model, ids, mem=None):
    """Log-prob of each token of ids (N, T) given the ones before it: (N, T-1)."""
    logits = model(ids[:, :-1], mem)[0].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def reward_exact_answer(completion, answer):
    """1 if the last number in the completion equals the reference answer."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", completion.replace(",", ""))
    return float(bool(nums) and nums[-1] == str(answer).strip())


def _boxed(text):
    """The content of the last \\boxed{...} in text (braces balanced), or None."""
    k = text.rfind("\\boxed{")
    if k < 0:
        return None
    depth, start = 0, k + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _same(a, b):
    """Two answers are the same if they are the same number, or the same text once spaces, $ and commas are gone."""
    norm = lambda t: re.sub(r"[\s$,]", "", str(t)).rstrip(".")
    a, b = norm(a), norm(b)
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b and a != ""


def reward_math(completion, answer):
    """1 if the last \\boxed{...} of the completion, or else its last number, equals the reference answer."""
    box = _boxed(completion)
    if box is not None:
        return float(_same(box, answer))
    nums = re.findall(r"-?\d+(?:\.\d+)?", completion.replace(",", ""))
    return float(bool(nums) and _same(nums[-1], answer))


def reward_exact(completion, answer):
    return float(completion.strip() == str(answer).strip())


def reward_code(completion, tests, timeout=5.0):
    """1 if the first Python block of the completion (or all of it), followed by `tests`, runs and exits with 0.
    Runs in its own process and an empty folder, isolated from the environment (-I), with a time limit and, on
    POSIX, a memory limit. That is not a sandbox: run it only in a container or a throwaway VM."""
    m = re.search(r"```(?:python|py)?\n(.*?)```", completion, re.S)
    code = (m.group(1) if m else completion) + "\n\n" + tests + "\n"

    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        except (ValueError, OSError):
            pass                                           # macOS refuses address-space limits
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=limits if os.name == "posix" else None, env={})
        except subprocess.TimeoutExpired:
            return 0.0
    return float(r.returncode == 0)


REWARDS = dict(math=reward_math, exact=reward_exact, code=reward_code)


def load_tasks(path):
    """[(messages, answer, reward name)] from a .jsonl or .jsonl.gz file; unreadable lines are skipped and counted."""
    out, skipped = [], 0
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                msgs = ([dict(role=m["role"], content=m["content"]) for m in rec["messages"]] if "messages" in rec
                        else [dict(role="user", content=rec["prompt"])])
                answer, name = rec["answer"], rec.get("reward", "math")
                ok = bool(msgs) and all(isinstance(m["content"], str) for m in msgs) and isinstance(answer, str) \
                    and answer.strip() != "" and name in REWARDS
            except (KeyError, TypeError, ValueError):
                ok = False
            if ok:
                out.append((msgs, answer, name))
            else:
                skipped += 1
    return out, skipped


def sample_group(model, tok, task, group, max_new_tokens, temperature, generator=None):
    """`group` sampled answers to one task. Returns [(ids, mask, text)]: ids is the prompt, the answer and, if the
    answer ended by itself, <|im_end|>; mask is 1 on the answer's tokens (the ending included)."""
    from ..chat import chat_prompt
    from ..generate import generate
    end = tok.token_to_id("<|im_end|>")
    prompt = chat_prompt(task[0], tok)
    was_training = model.training
    model.eval()
    new = generate(model, [prompt] * group, max_new_tokens, temperature=temperature, stop=[end], pad=end,
                   generator=generator)
    model.train(was_training)
    out = []
    for c in new:
        scored = c + [end] if len(c) < max_new_tokens else c       # shorter than the limit: it stopped by itself
        out.append((prompt + scored, [0] * len(prompt) + [1] * len(scored), tok.decode(c)))
    return out


@torch.no_grad()
def evaluate(model, tok, tasks, max_new_tokens, batch=16):
    """Mean reward of the most likely answer (temperature 0) to each task."""
    from ..chat import chat_prompt
    from ..generate import generate
    end = tok.token_to_id("<|im_end|>")
    was_training = model.training
    model.eval()
    total, by_len = 0.0, {}
    for t in tasks:                                        # prompts of one length share a call, so nothing is padded
        p = chat_prompt(t[0], tok)                         # (the model reads left padding as text)
        by_len.setdefault(len(p), []).append((p, t))
    for group in by_len.values():
        for b in range(0, len(group), batch):
            part = group[b:b + batch]
            new = generate(model, [p for p, _ in part], max_new_tokens, temperature=0, stop=[end], pad=end)
            total += sum(REWARDS[t[2]](tok.decode(c), t[1]) for (_, t), c in zip(part, new))
    model.train(was_training)
    return total / len(tasks)


def main(argv=None):
    from ..hub import build_model
    from ..model import ExeaLM, count_params
    from .pretrain import make_optimizer, pick_device, save
    from .sft import ROLE, pad_batch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the checkpoint to start from; a frozen copy is the reference")
    ap.add_argument("--tokenizer", default="tokenizer", help="the tokenizer the checkpoint was trained with")
    ap.add_argument("--data", required=True, help="tasks, .jsonl or .jsonl.gz")
    ap.add_argument("--val", default=None, help="held-out tasks (default: the last --val_frac of --data)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--prompts", type=int, default=8, help="tasks per step")
    ap.add_argument("--group", type=int, default=8, help="sampled answers per task")
    ap.add_argument("--epochs", type=int, default=2, help="optimizer steps on each batch of sampled answers")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--clip_ratio", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.04, help="weight of the KL term to the reference model")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--allow_code_exec", action="store_true", help="let the \"code\" reward run model-written code")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/grpo")
    a = ap.parse_args(argv)

    from ..data.tokenizer import load
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = pick_device(a.device)
    tok = load(a.tokenizer)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck)
    if tok.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError(f"the tokenizer has {tok.get_vocab_size()} tokens but the checkpoint was trained with "
                         f"{model.cfg.vocab_size}")
    for t in ("<|im_start|>", "<|im_end|>", "<|pad|>", *ROLE.values()):
        if tok.token_to_id(t) is None:
            raise ValueError(f"the tokenizer has no {t} token (train it with exealm.data.tokenizer)")
    pad = tok.token_to_id("<|pad|>")
    tasks, skipped = load_tasks(a.data)
    if any(t[2] == "code" for t in tasks) and not a.allow_code_exec:
        raise ValueError("these tasks use the \"code\" reward, which runs model-written code: pass --allow_code_exec, "
                         "inside a container or a throwaway VM only")
    if a.val:
        train, val = tasks, load_tasks(a.val)[0]
    else:
        n_val = max(1, int(len(tasks) * a.val_frac))
        train, val = tasks[:-n_val], tasks[-n_val:]
    if not train or not val:
        raise ValueError("no usable tasks: each needs a prompt, an answer and a known reward")
    model = model.to(dev)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.train()
    kind = "tonn" if isinstance(model, ExeaLM) else "llama"
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit(dict(event="start", model=kind, params=count_params(model), device=str(dev), tasks=len(train),
              val_tasks=len(val), skipped_lines=skipped, val_reward=round(evaluate(model, tok, val, a.max_new_tokens), 4)))
    opt = make_optimizer(model, a.lr, a.wd)
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    seen, t0 = [], time.time()
    for step in range(a.steps):
        rows, rewards = [], []
        for k in rng.choice(len(train), size=min(a.prompts, len(train)), replace=False):
            task = train[int(k)]
            for ids, mask, text in sample_group(model, tok, task, a.group, a.max_new_tokens, a.temperature, gen):
                rows.append((ids, mask))
                rewards.append(REWARDS[task[2]](text, task[1]))
        ids, mask = pad_batch(rows, pad)
        ids, mask = ids.to(dev), mask.to(dev)[:, 1:]
        adv = group_advantages(torch.tensor(rewards), a.group).to(dev)
        with torch.no_grad():
            logp_old, logp_ref = token_logprobs(model, ids), token_logprobs(ref, ids)
        for _ in range(a.epochs):
            with torch.autocast(**amp) if amp else torch.enable_grad():
                loss = grpo_loss(token_logprobs(model, ids), logp_old, logp_ref, adv, mask, a.clip_ratio, a.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
        seen.append((sum(rewards) / len(rewards), float(mask.sum(1).float().mean())))
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            emit(dict(event="eval", step=step + 1, sampled_reward=round(sum(r for r, _ in seen) / len(seen), 4),
                      answer_tokens=round(sum(n for _, n in seen) / len(seen), 1),
                      val_reward=round(evaluate(model, tok, val, a.max_new_tokens), 4), secs=round(time.time() - t0, 1)))
            seen = []
    save(os.path.join(a.out, "model.pt"), model, opt, a.steps, model.cfg, kind)
    return model


if __name__ == "__main__":
    main()
