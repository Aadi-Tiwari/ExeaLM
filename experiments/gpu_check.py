"""Does ExeaLM compute the same thing on this GPU as on the CPU? Run it once on every new kind of GPU, before trusting
a number from it. A few minutes on any recent NVIDIA card:

    python experiments/gpu_check.py

1. Layers. A TONN layer of each circuit type, an attention block and a Gated DeltaNet block, forward and backward:
   the GPU in 32-bit against the CPU (must agree to 1e-3 of the largest value), and the GPU in bf16 autocast against
   the GPU in 32-bit (must agree to 0.1: bf16 keeps 8 bits, so this only catches gross errors and non-finite values).
2. Whole models, with a swarm sharing a board: the loss and the size of its gradient, the same two comparisons.
3. Training: 150 steps on the made-up corpus with the search and a datastore. The loss must fall.
4. Resume: is a stopped and resumed run identical to the last bit? On a CPU it is. On a GPU it may not be (some
   kernels add in a different order from run to run), so this line is reported and never fails the check.
5. Two processes training one model (on two GPUs if there are two, else sharing one): they must stay in sync, and
   their resumed run is compared the same way.

Prints one line per check and exits with an error if a required one failed.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import replace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import ExeaLM, Llama, MemoryContext, get_config  # noqa: E402
from exealm.model.baseline import Attention  # noqa: E402
from exealm.model.circuits import CIRCUITS, MEMORY_CIRCUITS  # noqa: E402
from exealm.model.delta import GatedDeltaNet  # noqa: E402
from exealm.model.tonn import TONNLayer  # noqa: E402
from exealm.train.pretrain import load_model, main as pretrain, pick_device  # noqa: E402

FAILED = []


def report(name, ok, detail, required=True):
    print(f"{'ok    ' if ok else 'FAILED' if required else 'note  '} {name}: {detail}", flush=True)
    if required and not ok:
        FAILED.append(name)


def gap(a, b):
    """Largest difference as a share of the largest value. inf if either holds a NaN or an infinity."""
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("inf")
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def run(fn, x, dev, bf16=False):
    """fn's output for x on dev and the gradient of its squared mean with respect to x."""
    x = x.to(dev).requires_grad_(True)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if bf16 else torch.enable_grad():
        y = fn(x)
    y.float().pow(2).mean().backward()
    return y, x.grad


def compare(name, make, x, dev):
    """make(): the same freshly seeded module every time it is called."""
    torch.manual_seed(0)
    cpu = run(make().eval(), x, torch.device("cpu"))
    torch.manual_seed(0)
    mod = make().eval().to(dev)
    f32, b16 = run(mod, x, dev), run(mod, x, dev, bf16=True)
    a, b = max(gap(f32[0], cpu[0]), gap(f32[1], cpu[1])), max(gap(b16[0], f32[0]), gap(b16[1], f32[1]))
    report(name, a < 1e-3 and b < 0.1, f"32-bit against the CPU {a:.1e}, bf16 against 32-bit {b:.1e}")


def layers(dev):
    cfg = get_config("small-tonn")
    x = torch.randn(4, 256, cfg.d, generator=torch.Generator().manual_seed(1))
    for t in CIRCUITS:
        if t not in MEMORY_CIRCUITS:
            compare(f"circuit {t}", lambda: TONNLayer(cfg, [dict(tmpl=t, lag=3, lam0=0.5)] * cfg.slots, 0), x, dev)
    compare("attention block", lambda: Attention(cfg.d, cfg.d // 64, gate=True, qk_norm=True), x, dev)
    compare("Gated DeltaNet block", lambda: GatedDeltaNet(cfg.d, 64), x, dev)


def models(dev):
    cfg = get_config("small", vocab_size=4096)
    idx = torch.randint(0, cfg.vocab_size, (8, 257), generator=torch.Generator().manual_seed(2))
    for name, make in (("ExeaLM, swarms of 4", lambda: ExeaLM(cfg)), ("Llama", lambda: Llama(cfg)),
                       ("Qwen3.5-style", lambda: Llama(replace(cfg, delta_layers=(0, 1, 2, 4))))):
        out = []
        for d, bf16 in ((torch.device("cpu"), False), (dev, False), (dev, True)):
            torch.manual_seed(0)
            m = make().eval().to(d)
            with torch.autocast(device_type=d.type, dtype=torch.bfloat16) if bf16 else torch.enable_grad():
                loss = m(idx[:, :-1].to(d), MemoryContext(swarm_size=4), idx[:, 1:].to(d))[1]
            loss.backward()
            grad = torch.stack([p.grad.float().norm() for p in m.parameters() if p.grad is not None]).norm()
            out.append(torch.stack([loss.detach().float().cpu(), grad.cpu()]))
        a, b = gap(out[1], out[0]), gap(out[2], out[1])
        report(f"model {name}", a < 1e-3 and b < 0.1, f"loss {float(out[1][0]):.4f}; 32-bit against the CPU {a:.1e}, "
               f"bf16 against 32-bit {b:.1e}")


def identical(a, b):
    ma, mb = load_model(a)[0], load_model(b)[0]
    pairs = zip(ma.state_dict().values(), mb.state_dict().values())
    return ma.specs == mb.specs and all(torch.equal(v, w) for v, w in pairs)


def training(dev, tmp):
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "150", "--eval_every", "50", "--search_every", "30",
              "--datastore_chunks", "500", "--refresh_every", "40", "--device", str(dev)]
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    with open(os.devnull, "w") as quiet:
        stdout, sys.stdout = sys.stdout, quiet
        try:
            pretrain(common + ["--out", a])
            pretrain(common + ["--out", b, "--stop_at", "70"])
            pretrain(common + ["--out", b, "--resume", "auto"])
        finally:
            sys.stdout = stdout
    bits = [json.loads(l)["val_bits_per_token"] for l in open(os.path.join(a, "log.jsonl")) if '"eval"' in l]
    report("training", bits[-1] < bits[0] - 1.0, f"validation bits per token {bits[0]} -> {bits[-1]} in 150 steps")
    same = identical(os.path.join(a, "model.pt"), os.path.join(b, "model.pt"))
    report("resume", same, "identical to the last bit" if same else "trains on, but not bit for bit on this GPU",
           required=False)


def two_processes(dev, tmp):
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "60", "--eval_every", "20", "--search_every", "10",
              "--datastore_chunks", "300", "--refresh_every", "25", "--device", dev.type]

    def launch(extra):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", str(port),
               "-m", "exealm.train.pretrain"] + common + extra
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if r.returncode:
            print(r.stderr[-3000:])
        return r.returncode == 0

    a, b = os.path.join(tmp, "a2"), os.path.join(tmp, "b2")
    if not (launch(["--out", a]) and launch(["--out", b, "--stop_at", "30"]) and launch(["--out", b, "--resume", "auto"])):
        return report("two processes", False, "a run failed (its last output is above)")
    ev = [json.loads(l) for l in open(os.path.join(a, "log.jsonl"))]
    evals = [e for e in ev if e["event"] == "eval"]
    synced, bits = [e["processes_in_sync"] for e in evals], [e["val_bits_per_token"] for e in evals]
    report("two processes", all(synced) and bits[-1] < bits[0],
           f"on {min(2, torch.cuda.device_count()) if dev.type == 'cuda' else 0} GPU(s); in sync at every evaluation: "
           f"{synced}; bits {bits[0]} -> {bits[-1]}")
    same = identical(os.path.join(a, "model.pt"), os.path.join(b, "model.pt"))
    report("two processes, resume", same, "identical to the last bit" if same else "trains on, but not bit for bit",
           required=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip", default="", help="comma-separated parts to leave out: layers, models, training, two")
    a = ap.parse_args()
    dev = pick_device(a.device)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    print(f"device: {name}, torch {torch.__version__}" + (f", CUDA {torch.version.cuda}" if dev.type == "cuda" else ""))
    skip = set(a.skip.split(","))
    with tempfile.TemporaryDirectory() as tmp:
        for part, fn in (("layers", lambda: layers(dev)), ("models", lambda: models(dev)),
                         ("training", lambda: training(dev, tmp)), ("two", lambda: two_processes(dev, tmp))):
            if part not in skip and not (part == "two" and dev.type == "mps"):
                fn()
    print("ALL REQUIRED CHECKS PASSED" if not FAILED else f"FAILED: {', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
