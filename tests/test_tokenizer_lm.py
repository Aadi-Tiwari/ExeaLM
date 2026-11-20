"""experiments/tokenizer_lm.py: a batch run as micro-batches (--accum) gives the same loss and gradient as the whole batch."""
import copy
import importlib.util
import pathlib

import pytest
import torch

from exealm import ExeaLM, Llama, MemoryContext, get_config
from exealm.train.pretrain import make_optimizer

path = pathlib.Path(__file__).parents[1] / "experiments" / "tokenizer_lm.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("tokenizer_lm", path)
tokenizer_lm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_lm)


@pytest.mark.parametrize("accum", [1, 2, 4])
@pytest.mark.parametrize("arch", ["llama", "tonn"])
def test_train_step_takes_the_whole_batch_loss_and_gradient(arch, accum, monkeypatch):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)   # clipping would hide a scale error
    torch.manual_seed(0)
    model = Llama(get_config("tiny")) if arch == "llama" else ExeaLM(get_config("tiny"))
    if arch == "tonn":
        model.eval()          # in training mode the running slot scale updates once per micro-batch (see train_step)
    x, y = torch.randint(0, 256, (4, 16)), torch.randint(0, 256, (4, 16))
    ref = copy.deepcopy(model)
    want_loss = ref(x, MemoryContext(), y)[1]
    want_loss.backward()
    want = dict(ref.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in want.values())
    for p in model.parameters():
        p.grad = torch.ones_like(p)                     # stale gradients that the step must clear first
    loss = tokenizer_lm.train_step(model, make_optimizer(model, 1e-2, 0.1), x, y, torch.device("cpu"), accum)
    assert loss == pytest.approx(float(want_loss.detach()), abs=1e-6)
    for name, p in model.named_parameters():
        if want[name].grad is None:
            assert p.grad is None, name
        else:
            assert p.grad is not None and torch.allclose(p.grad, want[name].grad, rtol=1e-4, atol=1e-7), name


def test_train_step_rejects_a_batch_that_does_not_split_evenly():
    model = Llama(get_config("tiny"))
    x = torch.randint(0, 256, (6, 16))
    with pytest.raises(ValueError):
        tokenizer_lm.train_step(model, make_optimizer(model, 1e-2, 0.1), x, x, torch.device("cpu"), 4)
