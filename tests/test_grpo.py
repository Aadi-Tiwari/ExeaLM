"""GRPO: rewards, the task loader, sampling a group, and the loop."""
import gzip
import json

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from exealm import ExeaLM, get_config
from exealm.train import grpo, sft
from exealm.train.pretrain import make_optimizer, save

WORDS = ["red", "blue", "green", "black", "white", "pink"]
GOOD = dict(zip(WORDS, ["apple", "sky", "leaf", "night", "snow", "rose"]))
BAD = dict(zip(WORDS, ["sky", "leaf", "night", "snow", "rose", "apple"]))


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    with gzip.open(d / "a.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps(dict(text=" ".join(WORDS + list(GOOD.values())) + f" What is {i}?")) + "\n")
    return train([str(d / "a.jsonl.gz")], 330, str(d / "tokenizer")), str(d / "tokenizer")


def test_rewards():
    assert grpo.reward_math(r"first \boxed{\frac{2}{3}}, then \boxed{2x-8}.", "2x-8") == 1.0
    assert grpo.reward_math(r"\boxed{7}", "8") == 0.0
    assert grpo.reward_math("so the total is 1,024.0 apples", "1024") == 1.0
    assert grpo.reward_math("no number here", "3") == 0.0
    assert grpo.reward_exact(" sky \n", "sky") == 1.0 and grpo.reward_exact("the sky", "sky") == 0.0
    assert grpo.reward_code("```python\ndef add(a, b):\n    return a + b\n```\nDone.", "assert add(2, 3) == 5") == 1.0
    assert grpo.reward_code("def add(a, b):\n    return a - b", "assert add(2, 3) == 5") == 0.0
    assert grpo.reward_code("while True:\n    pass", "pass", timeout=1.0) == 0.0


def test_loader_reads_both_forms_and_skips_broken_lines(tmp_path):
    lines = [dict(prompt="2+2?", answer="4"), dict(messages=[dict(role="user", content="q")], answer="a", reward="exact"),
             dict(prompt="q", answer="a", reward="unknown"), dict(prompt="q"), dict(prompt="q", answer="")]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    tasks, skipped = grpo.load_tasks(str(tmp_path / "t.jsonl"))
    assert skipped == 4 and [t[2] for t in tasks] == ["math", "exact"]


def test_a_sampled_group_scores_the_answer_and_its_ending(tok):
    t, _ = tok
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1))
    task = ([dict(role="user", content="red")], "apple", "exact")
    rows = grpo.sample_group(model, t, task, 5, max_new_tokens=6, temperature=1.0)
    end = t.token_to_id("<|im_end|>")
    assert len(rows) == 5 and model.training
    for ids, mask, text in rows:
        n = sum(mask)
        assert len(ids) == len(mask) and mask[:len(mask) - n].count(1) == 0 and 1 <= n <= 6
        assert (ids[-1] == end) == (n < 6 or ids[-1] == end and len(t.encode(text).ids) < 6)
        assert end not in ids[len(ids) - n:-1]                              # the ending appears only as the last token


def test_code_rewards_are_refused_without_the_flag(tmp_path, tok):
    _, tok_dir = tok
    cfg = get_config("tiny", vocab_size=330, swarm_size=1)
    model = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), model, make_optimizer(model, 1e-3, 0.1), 0, cfg, "tonn")
    (tmp_path / "t.jsonl").write_text(json.dumps(dict(prompt="write add", answer="assert add(1, 1) == 2", reward="code")) + "\n")
    with pytest.raises(ValueError, match="allow_code_exec"):
        grpo.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "t.jsonl")])


def test_grpo_fixes_a_model_whose_most_likely_answer_is_wrong(tmp_path, tok):
    """SFT on data that gives the wrong reply two times in three, then GRPO with a reward for the right one."""
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1)
    base = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), base, make_optimizer(base, 1e-3, 0.1), 0, cfg, "tonn")
    rng = np.random.default_rng(0)
    with open(tmp_path / "sft.jsonl", "w") as f:
        for k in range(360):
            w = WORDS[k % len(WORDS)]
            reply = GOOD[w] if rng.random() < 0.35 else BAD[w]
            f.write(json.dumps(dict(messages=[dict(role="user", content=w), dict(role="assistant", content=reply)])) + "\n")
    sft.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "sft.jsonl"),
              "--steps", "200", "--batch", "16", "--lr", "3e-3", "--eval_every", "200", "--device", "cpu",
              "--out", str(tmp_path / "sft")])
    with open(tmp_path / "tasks.jsonl", "w") as f:
        for k in range(60):
            w = WORDS[k % len(WORDS)]
            f.write(json.dumps(dict(prompt=w, answer=GOOD[w], reward="exact")) + "\n")
    out = tmp_path / "run"
    grpo.main(["--ckpt", str(tmp_path / "sft" / "model.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "tasks.jsonl"),
               "--val_frac", "0.1", "--steps", "60", "--prompts", "6", "--group", "16", "--epochs", "2",
               "--max_new_tokens", "6", "--lr", "1e-3", "--beta", "0.01", "--eval_every", "20", "--device", "cpu",
               "--out", str(out)])
    log = [json.loads(line) for line in open(out / "log.jsonl")]
    assert log[0]["event"] == "start" and log[0]["tasks"] == 54 and log[0]["val_tasks"] == 6
    assert log[0]["val_reward"] <= 0.2                                      # the wrong reply is the likely one
    assert log[-1]["step"] == 60 and log[-1]["val_reward"] >= 0.8
    assert log[-1]["sampled_reward"] > log[1]["sampled_reward"] > 0.2
