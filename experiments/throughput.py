"""Training throughput on one GPU, measured the way results/funding/throughput_mi300x.json was, so the compute plan can
be rescaled to the machine the hours are on.

    python experiments/throughput.py --out results/expanse                     # 0.1B and 0.5B, both models, bf16 and 32-bit
    python experiments/throughput.py --sizes 1B,3B --precision bf16            # the two larger shapes
    python experiments/throughput.py --sizes 0.5B --precision bf16 --compile 1 # what torch.compile buys
    python experiments/throughput.py --sizes 0.5B --precision bf16 --tf32 1    # what TF32 buys in the 32-bit parts

A row is one model at one size and precision: seconds per training step (forward, backward, gradient clip, optimizer)
on random tokens, tokens per second, effective TFLOP/s (6 x parameters x tokens per second) and the peak GPU memory.
"hybrid" is the size's preset without the memory circuits (as in the MI300X rows: 93.4M and 497.1M parameters with
the 32k tokenizer) and "llama" the plain same-size Llama. If a batch does not fit the GPU, it is halved and the row
says so.

For bf16 rows the script also prints what each run of the compute plan would cost at the measured rate, next to what
the plan assumed. These are single-GPU rates. For a 4-GPU run, time the real loop (experiments/expanse/README.md).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import ExeaLM, Llama, MemoryContext, count_params, get_config, llama_baseline  # noqa: E402
from exealm.config import MENU  # noqa: E402
from exealm.train.pretrain import make_optimizer, pick_device  # noqa: E402

PRESET = {"0.1B": "100m", "0.5B": "500m", "1B": "1b", "3B": "3b", "tiny": "tiny"}
BATCH = {"0.1B": 32, "0.5B": 16, "1B": 16, "3B": 8, "tiny": 8}            # 0.1B and 0.5B: as measured on the MI300X
# the compute plan (ExeaLM_nairr/6_compute_plan.tex): training tokens, and the GPU-hours it budgeted on MI300X rates
PLAN = {("0.5B", "hybrid"): (30e9, 307), ("0.5B", "llama"): (30e9, 190), ("1B", "hybrid"): (20e9, 397),
        ("1B", "llama"): (20e9, 248), ("3B", "hybrid"): (8e9, 477)}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def build(size, kind, vocab, dev):
    over = dict(vocab_size=vocab) if size != "tiny" else {}
    cfg = get_config(PRESET[size], pinned={}, menu=tuple(t for t in MENU if t not in ("RG", "AG")), **over)
    return (ExeaLM(cfg) if kind == "hybrid" else Llama(llama_baseline(cfg))).to(dev)


def measure(model, batch, a, dev, bf16):
    """Median seconds per step over a.steps steps, after a.warmup steps that are not timed."""
    opt = make_optimizer(model, 1e-4, 0.1, fused=dev.type == "cuda")
    fwd = torch.compile(model) if a.compile else model
    x = torch.randint(0, model.cfg.vocab_size, (batch, a.seq + 1), device=dev)
    times = []
    for i in range(a.warmup + a.steps):
        sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if bf16 else torch.enable_grad():
            loss = fwd(x[:, :-1], MemoryContext(), x[:, 1:])[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(dev)
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def row(size, kind, precision, a, dev):
    batch = a.batch or BATCH[size]
    rec = dict(size=size, model=kind, precision=precision, seq=a.seq)
    while True:
        model = None
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            torch.manual_seed(0)
            model = build(size, kind, a.vocab, dev)
            s = measure(model, batch, a, dev, precision == "bf16")
            params = count_params(model)
            rec.update(params_M=round(params / 1e6, 1), batch=batch, s_per_step=round(s, 4),
                       tokens_per_s=round(batch * a.seq / s),
                       effective_TFLOPs=round(6 * params * batch * a.seq / s / 1e12, 1))
            if dev.type == "cuda":
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1)
            return rec
        except torch.cuda.OutOfMemoryError:
            del model
            rec["did_not_fit_batch"] = batch
            batch //= 2
            if batch == 0:
                return dict(rec, error="out of memory at batch 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.1B,0.5B", help="any of " + ", ".join(PRESET))
    ap.add_argument("--models", default="hybrid,llama")
    ap.add_argument("--precision", default="bf16,fp32")
    ap.add_argument("--batch", type=int, default=0, help="rows per step (default: 32, 16, 16 and 8 for the four sizes)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=20, help="timed steps")
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps first (use 20 or more with --compile 1)")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--tf32", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="folder for throughput_<tag>.json")
    ap.add_argument("--tag", default=None, help="name of the output file (default: from the GPU's name)")
    a = ap.parse_args()
    dev = pick_device(a.device)
    if a.tf32 and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    rows = []
    for size in a.sizes.split(","):
        for kind in a.models.split(","):
            for precision in a.precision.split(","):
                rows.append(dict(row(size, kind, precision, a, dev), gpu=gpu, torch=torch.__version__,
                                 compile=a.compile, tf32=a.tf32))
                print(json.dumps(rows[-1]), flush=True)
    plan = [(r, PLAN[(r["size"], r["model"])]) for r in rows
            if r["precision"] == "bf16" and "tokens_per_s" in r and (r["size"], r["model"]) in PLAN]
    if plan:
        print("\nThe compute plan at these single-GPU rates:")
        for r, (tokens, budget) in plan:
            print(f"  {r['size']} {r['model']}: {tokens / 1e9:.0f}B tokens = {tokens / r['tokens_per_s'] / 3600:,.0f} "
                  f"GPU-hours (the plan budgeted {budget})")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or "".join(c if c.isalnum() else "_" for c in gpu.lower()).strip("_")
        path = os.path.join(a.out, f"throughput_{tag}.json")
        old = json.load(open(path)) if os.path.exists(path) else []
        json.dump(old + rows, open(path, "w"), indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
