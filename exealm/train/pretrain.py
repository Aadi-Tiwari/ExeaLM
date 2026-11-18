"""Pretraining: next-token prediction, with TONN's architecture search early on, the retrieval index refreshed as the
model's own encoder learns, and every batch split into swarms of agents that share a board (docs/swarms.md).

Smoke test, no download needed (CPU, about a minute):
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --datastore_chunks 2000

The fair comparison every architecture change must win (same data, same steps, same parameter count):
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 2000 --model tonn --out runs/tonn
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 2000 --model llama --out runs/llama

On real data, after exealm.data.pack:
    python -m exealm.train.pretrain --preset 100m --data data/packed/train --val data/packed/val --seq 2048 --batch 16

A mix of sources with weights, each source in many packed directories (the form is in exealm/data/loader.py: open_data):
    python -m exealm.train.pretrain --preset 100m --data mix.json --val data/packed/val --datastore_data data/packed/library

Resuming: --save_every N writes the full training state to <out>/ckpt.pt every N steps (the model, the optimizer, the
current architecture, every random number generator, the search's state and the datastore's keys). Pick the run up
with --resume <out>/ckpt.pt, or --resume auto to use <out>/ckpt.pt if it exists. With the same arguments, a resumed run
ends with exactly the weights an uninterrupted run would have (tested on CPU). --stop_at N saves and stops after step N
without changing the learning-rate schedule, for sessions with a time limit such as Colab:
    python -m exealm.train.pretrain --preset small --data data/packed/train --steps 20000 --save_every 500 \
        --stop_at 5000 --resume auto      # run the same line again in the next session to continue

Several GPUs: start the same command with torchrun, one process per GPU (exealm/train/multi.py):
    torchrun --nproc_per_node 4 -m exealm.train.pretrain --preset 500m --data mix.json --batch 16 --accum 4 ...
--batch is the rows each GPU reads in one forward pass and --accum how many passes go into one optimizer step, so one
step trains on batch x seq x accum x GPUs tokens. --steps, --warmup, --eval_every and --save_every count optimizer
steps. Only process 0 logs, evaluates, searches and saves.

Time limits (Slurm and the like): --save_minutes M also writes <out>/ckpt.pt every M minutes, and --max_minutes M
saves and stops M minutes into this session, keeping the --steps schedule, so a job ends cleanly before its limit and
the next one continues with --resume auto. SIGUSR1 or SIGTERM does the same within a step of arriving.

On NVIDIA GPUs the loop uses bf16 autocast (--amp) and a fused AdamW. Two more switches are off until measured:
--tf32 1 lets the parts we keep in 32-bit for precision (scans, the memory circuits, Gated DeltaNet) multiply in
TF32, and --compile 1 runs the model through torch.compile once the search has frozen the architecture.
  TODO(infra): models that do not fit one GPU (FSDP); the datastore search sharded across GPUs.

--wandb PROJECT also sends every eval, search and resume record to Weights & Biases (pip install wandb). A resumed run
continues the same W&B run: its id is kept in <out>/wandb_id.txt.
"""
import argparse
import json
import math
import os
import signal
import threading
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from ..config import ModelConfig, get_config
from ..data.loader import TokenShards, open_data
from ..memory import Datastore, MemoryContext
from ..model import ExeaLM, Llama, count_params
from ..model.search import Searcher
from .multi import Group


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step, total, peak, warmup, decay_frac, floor=0.1):
    """Warmup-stable-decay (WSD): linear warmup, flat, then a linear decay over the last decay_frac of training.
    WSD lets us branch off the flat part to try a different final data mix without restarting (docs/training.md)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return peak
    return peak * (floor + (1 - floor) * (1 - (step - d0) / max(1, total - d0)))


def board_summary(usage):
    """usage: per layer, a list of exealm.model.board.usage_stats dicts (one per validation batch). Returns each stat
    averaged over everything, plus read_nothing and post_strength per layer (where the board is used differs a lot
    between layers)."""
    flat = [d for per_layer in usage for d in per_layer]
    out = {k: round(sum(d[k] for d in flat) / len(flat), 4) for k in flat[0]}
    for k in ("read_nothing", "post_strength"):
        out[k + "_per_layer"] = [round(sum(d[k] for d in u) / len(u), 4) if u else None for u in usage]
    return out


def wandb_record(rec, prefix=""):
    """A log record's numbers as flat "eval/val_bits_per_token"-style keys: W&B plots numbers, so nested dicts are
    flattened, lists become one key per item, and text and missing values are left out."""
    out = {}
    if not prefix:
        prefix = str(rec.get("event", "log")) + "/"
    for k, v in rec.items():
        if k in ("event", "step") and prefix.count("/") == 1:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[prefix + k] = v
        elif isinstance(v, dict):
            out.update(wandb_record(v, prefix + k + "/"))
        elif isinstance(v, (list, tuple)):
            out.update(wandb_record({str(i): x for i, x in enumerate(v)}, prefix + k + "/"))
    return out


def make_optimizer(model, lr, wd, fused=False):
    """fused: the whole AdamW step as one CUDA kernel per tensor. Faster, and it skips the temporaries of the default
    step, which matters once a model and its optimizer statistics fill most of a GPU."""
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    return torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=rest, weight_decay=0.0)],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, **(dict(fused=True) if fused else {}))


def save(path, model, opt, step, cfg, kind, state=None):
    """state: what else an exact resume needs (random generators, search, datastore keys, see main()). Written to a
    temporary file first, so a run killed mid-save never leaves a broken checkpoint behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=asdict(cfg), kind=kind,
                    specs=getattr(model, "specs", None), state=state), tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    model = ExeaLM(cfg, ck["specs"]) if ck["kind"] == "tonn" else Llama(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--config", default=None, help="JSON file of ModelConfig fields to override")
    ap.add_argument("--model", default="tonn", choices=["tonn", "llama"])
    ap.add_argument("--data", default="synthetic",
                    help="'synthetic', a packed directory, a pattern of packed directories, or a mix .json with "
                         "per-source weights (exealm/data/loader.py: open_data)")
    ap.add_argument("--val", default=None, help="validation split, same forms as --data (default: windows of --data)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="rows per forward pass, on each GPU")
    ap.add_argument("--accum", type=int, default=1,
                    help="forward passes per optimizer step (gradient accumulation), for batches that do not fit")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--swarm_policy", default="mixed", choices=["mixed", "split", "random"],
                    help="what each swarm reads (docs/swarms.md)")
    ap.add_argument("--search", type=int, default=1)
    ap.add_argument("--search_every", type=int, default=100)
    ap.add_argument("--search_until", type=float, default=0.5)
    ap.add_argument("--datastore_chunks", type=int, default=0, help="0: no retrieval (RG slots output zeros)")
    ap.add_argument("--datastore_data", default=None,
                    help="one packed directory to sample the retrieval library from (default: --data, which must "
                         "then be one packed directory)")
    ap.add_argument("--refresh_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "none"],
                    help="bfloat16 autocast: auto turns it on for CUDA and ROCm GPUs")
    ap.add_argument("--tf32", type=int, default=0,
                    help="1: 32-bit matrix products may use TF32 on NVIDIA GPUs (faster, fewer mantissa bits)")
    ap.add_argument("--compile", type=int, default=0,
                    help="1: torch.compile the model once the architecture no longer changes (after the search)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--save_every", type=int, default=0, help="write <out>/ckpt.pt every this many steps (0: never)")
    ap.add_argument("--save_minutes", type=float, default=0, help="also write it every this many minutes (0: never)")
    ap.add_argument("--max_minutes", type=float, default=0,
                    help="save <out>/ckpt.pt and stop this many minutes into the session, keeping the --steps "
                         "schedule (for job time limits; 0: no limit)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from, or 'auto' for <out>/ckpt.pt if it exists")
    ap.add_argument("--stop_at", type=int, default=0,
                    help="save <out>/ckpt.pt and stop after this step, keeping the --steps schedule (for time limits)")
    ap.add_argument("--wandb", default=None, help="Weights & Biases project to log to (default: no W&B logging)")
    ap.add_argument("--wandb_name", default=None, help="name of the W&B run (default: the --out folder's name)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    grp = Group(pick_device(a.device))                                  # under torchrun: one process per GPU
    dev = grp.device
    if a.tf32 and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = a.amp == "bf16" or (a.amp == "auto" and dev.type == "cuda")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16) if use_amp else None
    data = open_data(a.data)
    val = data if a.val is None else open_data(a.val)
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    resume = ckpt_path if a.resume == "auto" else a.resume
    if a.resume == "auto" and not os.path.exists(ckpt_path):
        resume = None
    ck = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if ck is not None:
        if ck["kind"] != a.model:
            raise ValueError(f"checkpoint is a {ck['kind']} model, but --model is {a.model}")
        if ck.get("state") is None:
            raise ValueError(f"{resume} has no training state to resume from (saved before --save_every existed)")
        cfg = ModelConfig.from_dict(ck["cfg"])                          # the run's own config and architecture
        model = (ExeaLM(cfg, ck["specs"]) if a.model == "tonn" else Llama(cfg)).to(dev)
        model.load_state_dict(ck["model"])
    else:
        over = json.load(open(a.config)) if a.config else {}
        over["vocab_size"] = int(data.meta["vocab_size"])
        over.setdefault("seed", a.seed)                                # --seed also draws the starting circuits
        if a.model == "llama":                                         # the baseline is a plain Llama unless the
            over.setdefault("attn_gate", False)                        # --config asks for the Qwen3.5 parts
            over.setdefault("qk_norm", False)
        cfg = get_config(a.preset, **over)
        model = (ExeaLM(cfg) if a.model == "tonn" else Llama(cfg)).to(dev)
    opt = make_optimizer(model, a.lr, a.wd, fused=dev.type == "cuda")
    if ck is not None:
        opt.load_state_dict(ck["opt"])
    grp.send_state(model, opt)                                          # every process starts from process 0's model
    warmup = a.warmup if a.warmup is not None else max(1, min(2000, a.steps // 20))
    swarm = cfg.swarm_size if a.model == "tonn" else 1
    rng = np.random.default_rng(a.seed if grp.lead else [a.seed, grp.rank])     # each process reads its own batches
    tokens_per_step = a.batch * a.seq * a.accum * grp.world

    ds = None
    if a.model == "tonn" and a.datastore_chunks:
        if a.datastore_data:
            ds = Datastore.from_shards(TokenShards(a.datastore_data), a.datastore_chunks, cfg.rag_chunk_len,
                                       seed=a.seed + 1)
            ds.doc_ids += int(data.meta["n_docs"])                     # library documents are not training documents
        elif hasattr(data, "tokens"):
            ds = Datastore.from_shards(data, a.datastore_chunks, cfg.rag_chunk_len, seed=a.seed + 1)
        else:
            raise ValueError("--data is a mix or has many shards: name the retrieval library with --datastore_data")
        ds = ds.to(dev)
        ds.refresh(model.encoder, model.emb, step=0)

    def mem_for(docs):
        return MemoryContext(datastore=ds, doc_ids=docs.to(dev), swarm_size=swarm)

    def fixed_batches(src, n, seed):
        r = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            x, y, d = src.batch(a.batch, a.seq, r, swarm, a.swarm_policy)
            out.append((x.to(dev), y.to(dev), mem_for(d)))
        return out

    val_batches = fixed_batches(val, a.eval_batches, 10_000) if grp.lead else []
    sel_batches = fixed_batches(data, a.eval_batches, 20_000) if grp.lead else []   # the search's own held-out windows
    searcher = Searcher(every=a.search_every, until=a.search_until, seed=a.seed) if (a.search and a.model == "tonn") \
        else None

    @torch.no_grad()
    def evaluate():
        """Validation bits. For swarms, also with the board cut (every agent alone): the difference is what the
        agents gained from each other."""
        model.eval()
        bpt = float(data.meta.get("bytes_per_token", 1.0))
        boards = [lay.ag for lay in model.layers] if a.model == "tonn" else []
        usage = [[] for _ in boards]                                     # per layer, per batch: board_usage stats
        loss = 0.0
        for b in boards:                                                 # recorded on the shared-board pass only
            b.record = True
        with torch.autocast(**amp) if amp else torch.no_grad():
            for x, y, m in val_batches:
                loss += float(model(x, m, y)[1]) / len(val_batches)
                for u, b in zip(usage, boards):
                    if b.last is not None:
                        u.append(b.last)
            for b in boards:
                b.record, b.last = False, None
            cut = None
            if swarm > 1:
                cut = sum(float(model(x, replace(m, swarm_size=1), y)[1]) for x, y, m in val_batches) / len(val_batches)
        model.train()
        bits = loss / math.log(2)
        out = dict(val_bits_per_token=round(bits, 4), val_bits_per_byte=round(bits / bpt, 4))
        if cut is not None:
            out["val_bits_per_token_board_cut"] = round(cut / math.log(2), 4)
            out["board_gain_bits_per_token"] = round((cut - loss) / math.log(2), 4)
        if any(usage):
            out["board"] = board_summary(usage)
        return out

    log = run = None
    if grp.lead:
        os.makedirs(a.out, exist_ok=True)
        log = open(os.path.join(a.out, "log.jsonl"), "a")
    if a.wandb and grp.lead:
        import wandb
        id_file = os.path.join(a.out, "wandb_id.txt")                    # so a resumed run continues the same W&B run
        run_id = open(id_file).read().strip() if resume and os.path.exists(id_file) else None
        run = wandb.init(project=a.wandb, name=a.wandb_name or os.path.basename(os.path.normpath(a.out)), id=run_id,
                         resume="allow", config=dict(vars(a), model_cfg=asdict(cfg), params=count_params(model),
                                                     gpus=grp.world, tokens_per_step=tokens_per_step))
        with open(id_file, "w") as f:
            f.write(str(run.id))

    def emit(rec):
        if not grp.lead:
            return
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if run is not None and "step" in rec:
            run.log(wandb_record(rec), step=rec["step"])

    def checkpoint(path, step, losses):
        """Write the full training state: the weights and the optimizer, and everything else the next step depends
        on. Every process reports the state of its own batch sampler; process 0 writes the file."""
        rngs = grp.gather(rng.bit_generator.state)
        if not grp.lead:
            return
        state = dict(numpy_rng=rngs[0], numpy_rngs=rngs, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                     searcher=None if searcher is None else dict(round=searcher.round, born=dict(searcher.born),
                                                                 rng=searcher.rng.getstate()),
                     datastore=None if ds is None else dict(keys=ds.keys.cpu(), refreshed_at=ds.refreshed_at),
                     losses=[float(l) for l in losses], secs=time.time() - t0, total_steps=a.steps)
        save(path, model, opt, step, cfg, a.model, state)

    start, losses, done_secs = 0, [], 0.0
    if ck is not None:
        st = ck["state"]
        start, done_secs = ck["step"], st["secs"]
        rngs = st.get("numpy_rngs") or [st["numpy_rng"]]
        if grp.rank < len(rngs):
            rng.bit_generator.state = rngs[grp.rank]
        else:                                              # more processes than the run was saved with: a new stream
            rng = np.random.default_rng([a.seed, grp.rank, start])
        torch.set_rng_state(st["torch_rng"])
        c = st.get("cuda_rng")                             # older checkpoints: a list, one state per visible GPU
        if c is not None and dev.type == "cuda":
            torch.cuda.set_rng_state(c[min(dev.index or 0, len(c) - 1)] if isinstance(c, (list, tuple)) else c, dev)
        if searcher is not None and st["searcher"] is not None:
            searcher.round, searcher.born = st["searcher"]["round"], dict(st["searcher"]["born"])
            searcher.rng.setstate(st["searcher"]["rng"])
        if ds is not None and st["datastore"] is not None:
            ds.keys, ds.refreshed_at = st["datastore"]["keys"].to(dev), st["datastore"]["refreshed_at"]
        losses = [torch.tensor(l, device=dev) for l in st["losses"]]
        emit(dict(event="resume", step=start, path=resume, total_steps=a.steps,
                  saved_total_steps=st["total_steps"], gpus=grp.world, saved_gpus=len(rngs)))
    else:
        emit(dict(event="start", model=a.model, params=count_params(model), device=str(dev), gpus=grp.world,
                  tokens_per_step=tokens_per_step, cfg=asdict(cfg),
                  architecture=model.architecture() if a.model == "tonn" else None))

    asked = dict(stop=False)

    def ask_to_stop(signum, frame):
        asked["stop"] = True
        signal.signal(signum, signal.SIG_DFL)                           # a second signal ends the process at once

    handlers = {}
    if threading.current_thread() is threading.main_thread():
        handlers = {s: signal.signal(s, ask_to_stop) for s in (getattr(signal, "SIGUSR1", None), signal.SIGTERM) if s}

    # the one piece of model state each process measures on its own batches (exealm/train/multi.py)
    scales = [lay.scale for lay in model.layers] if a.model == "tonn" else []
    fwd = model
    t0 = time.time() - done_secs
    began = last_save = tick = time.time()
    tick_step = start
    try:
        for step in range(start, a.steps):
            if searcher is not None and searcher.due(step, a.steps):
                rec = searcher.maybe_round(model, opt, step, a.steps, sel_batches, a.seq) if grp.lead else None
                if rec:
                    emit(dict(event="search", **rec))
                if grp.many:                                            # the other processes take over what changed
                    specs = grp.share(model.specs if rec and rec["replaced"] else None)
                    if specs is not None:
                        for lay, row in zip(model.layers, specs):
                            for k, s in enumerate(row):
                                if s != lay.specs[k]:
                                    lay.set_spec(k, s, reinit=False)
                        grp.send_state(model, opt)
            if a.compile and fwd is model and (searcher is None or step >= a.search_until * a.steps):
                fwd = torch.compile(model)                              # the architecture is fixed from here on
            if ds is not None and step > 0 and step % a.refresh_every == 0:
                ds.refresh(model.encoder, model.emb, step=step)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.lr, warmup, a.decay_frac)
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for _ in range(a.accum):
                x, y, d = data.batch(a.batch, a.seq, rng, swarm, a.swarm_policy)
                with torch.autocast(**amp) if amp else torch.enable_grad():
                    part = fwd(x.to(dev), mem_for(d), y.to(dev))[1] / a.accum
                part.backward()
                loss = loss + part.detach()
            want_save = a.save_minutes > 0 and time.time() - last_save >= 60 * a.save_minutes
            want_stop = asked["stop"] or (a.max_minutes > 0 and time.time() - began >= 60 * a.max_minutes)
            if grp.many:                                                # one gradient, one model, one decision
                grp.average_grads(model)
                grp.average_(scales)
                loss, n_save, n_stop = grp.total([loss, want_save, want_stop])
                loss, want_save, want_stop = (loss / grp.world).float(), bool(n_save > 0), bool(n_stop > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            losses.append(loss)
            n, last = step + 1, step + 1 == a.steps
            if n % a.eval_every == 0 or last:
                same = grp.in_sync(model)
                if not same:
                    grp.send_state(model, opt)
                if grp.lead:
                    # tokens per second since the last evaluation, with searches, refreshes and saves counted in
                    rec = dict(event="eval", step=n, train_loss=round(float(torch.stack(losses).mean()), 4),
                               lr=round(opt.param_groups[0]["lr"], 6), secs=round(time.time() - t0, 1),
                               tokens=n * tokens_per_step,
                               tokens_per_sec=round((n - tick_step) * tokens_per_step / max(time.time() - tick, 1e-9)))
                    if dev.type == "cuda":
                        rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1)
                    if grp.many:
                        rec["processes_in_sync"] = same
                    emit(dict(rec, **evaluate()))
                losses = []
                tick, tick_step = time.time(), n
            stop = not last and (n == a.stop_at or want_stop)
            if not last and (stop or want_save or (a.save_every and n % a.save_every == 0)):
                checkpoint(ckpt_path, n, losses)
                last_save = time.time()
            if stop:
                emit(dict(event="stop", step=n, ckpt=ckpt_path,
                          why="stop_at" if n == a.stop_at else "signal" if asked["stop"] else "max_minutes"))
                return model
        if a.model == "tonn":
            emit(dict(event="end", architecture=model.architecture()))
        checkpoint(os.path.join(a.out, "model.pt"), a.steps, losses)
        return model
    finally:
        for s, h in handlers.items():
            if h is not None:
                signal.signal(s, h)
        if run is not None:
            run.finish()
        grp.close()


if __name__ == "__main__":
    main()
