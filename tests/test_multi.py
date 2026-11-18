"""Training with several processes (exealm/train/multi.py), on a CPU with two of them, and the options added for
long runs on shared GPUs: gradient accumulation, time limits and stop signals."""
import json
import os
import signal
import socket
import subprocess
import sys

import pytest
import torch

from exealm.train import pretrain
from exealm.train.pretrain import load_model, main

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = ["--preset", "tiny", "--data", "synthetic", "--device", "cpu"]
pytestmark = pytest.mark.skipif(not torch.distributed.is_available(), reason="this PyTorch has no torch.distributed")


def torchrun(args, processes=2):
    with socket.socket() as s:                                            # a free port, so runs never collide
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(processes), "--master_port",
           str(port), os.path.join(HERE, "multi_driver.py")] + [str(x) for x in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=os.path.dirname(HERE))
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-4000:])


def events(folder):
    return [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines()]


def same_weights(a, b):
    for (k, v), w in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(v, w), k


def test_gradients_are_averaged_over_the_processes(tmp_path):
    torchrun(["grads", tmp_path / "grads.json"])
    r = json.loads((tmp_path / "grads.json").read_text())
    assert r["processes"] == 2 and r["worst"] < 1e-6
    assert r["unused"] > 0 and r["partly"] == 1                           # both special cases were exercised


def test_two_processes_train_one_model_and_resume_exactly(tmp_path):
    common = TINY + ["--steps", "40", "--eval_every", "20", "--search_every", "4", "--search_until", "0.9",
                     "--datastore_chunks", "300", "--refresh_every", "15"]
    a, b = tmp_path / "a", tmp_path / "b"
    torchrun(["train", "replace", "--"] + common + ["--out", a])                       # straight through
    ev = events(a)
    assert ev[0]["gpus"] == 2 and ev[0]["tokens_per_step"] == 2 * 8 * 128
    evals = [e for e in ev if e["event"] == "eval"]
    assert evals[-1]["val_bits_per_token"] < evals[0]["val_bits_per_token"] < 8.0
    assert all(e["processes_in_sync"] for e in evals)                     # also after rounds that replaced circuits
    assert any(e["replaced"] > 0 for e in ev if e["event"] == "search")
    assert evals[-1]["tokens"] == 40 * 2 * 8 * 128

    torchrun(["train", "replace", "--"] + common + ["--out", b, "--stop_at", "6"])     # between two such rounds
    torchrun(["train", "replace", "--"] + common + ["--out", b, "--resume", "auto"])
    full, _ = load_model(str(a / "model.pt"))
    resumed, ck = load_model(str(b / "model.pt"))
    assert ck["step"] == 40 and len(ck["state"]["numpy_rngs"]) == 2       # one batch sampler per process
    assert [(e["step"], e["saved_gpus"]) for e in events(b) if e["event"] == "resume"] == [(6, 2)]
    assert full.specs == resumed.specs
    same_weights(full, resumed)


def test_processes_that_drift_apart_are_noticed(tmp_path):
    torchrun(["train", "noscale", "--"] + TINY + ["--steps", "10", "--eval_every", "5", "--search", "0",
                                                  "--out", tmp_path])
    assert not any(e["processes_in_sync"] for e in events(tmp_path) if e["event"] == "eval")


def test_accumulated_steps_match_one_larger_batch(tmp_path):
    """A Llama has no swarms and no running statistics, so 2 passes of 4 rows must give the step of 1 pass of 8."""
    common = TINY + ["--model", "llama", "--steps", "5", "--eval_every", "5"]
    main(common + ["--batch", "8", "--out", str(tmp_path / "one")])
    main(common + ["--batch", "4", "--accum", "2", "--out", str(tmp_path / "two")])
    assert events(tmp_path / "two")[0]["tokens_per_step"] == events(tmp_path / "one")[0]["tokens_per_step"] == 8 * 128
    one, two = load_model(str(tmp_path / "one" / "model.pt"))[0], load_model(str(tmp_path / "two" / "model.pt"))[0]
    for (k, v), w in zip(one.state_dict().items(), two.state_dict().values()):
        assert torch.allclose(v, w, atol=1e-4), k


def test_a_time_limit_stops_cleanly_and_the_next_session_finishes_the_same_run(tmp_path):
    common = TINY + ["--steps", "12", "--eval_every", "6", "--search_every", "4", "--datastore_chunks", "200"]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])
    main(common + ["--out", str(b), "--max_minutes", "1e-9"])             # the limit is over after the first step
    stop = [e for e in events(b) if e["event"] == "stop"]
    assert [(e["step"], e["why"]) for e in stop] == [(1, "max_minutes")] and (b / "ckpt.pt").exists()
    main(common + ["--out", str(b), "--resume", "auto"])
    same_weights(load_model(str(a / "model.pt"))[0], load_model(str(b / "model.pt"))[0])


def test_a_stop_signal_saves_and_stops_within_a_step(tmp_path, monkeypatch):
    """The signal arrives late in step 3, after that step's decision to go on, so the run stops after step 4."""
    real, calls = torch.nn.utils.clip_grad_norm_, []

    def clip_and_signal(*args, **kwargs):                                 # called once per step, inside the loop
        calls.append(1)
        if len(calls) == 3:
            signal.raise_signal(signal.SIGUSR1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pretrain.torch.nn.utils, "clip_grad_norm_", clip_and_signal)
    before = signal.getsignal(signal.SIGUSR1)
    main(TINY + ["--steps", "20", "--eval_every", "10", "--out", str(tmp_path)])
    assert signal.getsignal(signal.SIGUSR1) == before                     # the run put the old handler back
    stop = [e for e in events(tmp_path) if e["event"] == "stop"]
    assert [(e["step"], e["why"]) for e in stop] == [(4, "signal")]
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["step"] == 4


def test_save_minutes_writes_checkpoints_on_the_clock(tmp_path):
    main(TINY + ["--steps", "4", "--eval_every", "4", "--save_minutes", "1e-9", "--out", str(tmp_path)])
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["step"] == 3       # the last step writes model.pt
