"""experiments/tokenizer_lm.py: a batch run as micro-batches (--accum) gives the same loss and gradient as the whole batch."""
import copy
import importlib.util
import pathlib

import pytest
import torch

from exealm import ExeaLM, Llama, get_config
from exealm.train.pretrain import make_optimizer

path = pathlib.Path(__file__).parents[1] / "experiments" / "tokenizer_lm.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("tokenizer_lm", path)
tokenizer_lm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_lm)


@pytest.mark.parametrize("arch", ["llama", "tonn"])
def test_micro_batches_give_the_whole_batch_loss_and_gradient(arch, monkeypatch):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)   # clipping would hide a scale error
    torch.manual_seed(0)
    whole = Llama(get_config("tiny")) if arch == "llama" else ExeaLM(get_config("tiny"))
    if arch == "tonn":
        whole.eval()          # in training mode the running slot scale updates once per micro-batch (see train_step)
    split = copy.deepcopy(whole)
    x, y = torch.randint(0, 256, (4, 16)), torch.randint(0, 256, (4, 16))
    cpu = torch.device("cpu")
    loss_whole = tokenizer_lm.train_step(whole, make_optimizer(whole, 1e-2, 0.1), x, y, cpu, accum=1)
    loss_split = tokenizer_lm.train_step(split, make_optimizer(split, 1e-2, 0.1), x, y, cpu, accum=2)
    assert loss_split == pytest.approx(loss_whole, abs=1e-6)
    for (name, p), q in zip(whole.named_parameters(), split.parameters()):
        if p.grad is None:
            assert q.grad is None, name
            continue
        assert torch.allclose(q.grad, p.grad, rtol=1e-4, atol=1e-7), name
