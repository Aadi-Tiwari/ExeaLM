"""What tests/test_multi.py starts under torchrun, once per process. Not a test file itself.

    multi_driver.py train [replace] [noscale] -- <arguments of exealm.train.pretrain>
    multi_driver.py grads <output json>

"replace" makes every search round find all circuits worthless, so rounds replace circuits whatever this platform's
tiny model happens to learn. "noscale" switches off the averaging of the slots' running sizes, to show that the
processes then drift apart and that the check notices.
"""
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import ExeaLM, MemoryContext, get_config  # noqa: E402
from exealm.model import search  # noqa: E402
from exealm.train import multi, pretrain  # noqa: E402


def train(flags, argv):
    if "replace" in flags:
        real = search.group_worth

        def worthless(model, batches):
            worth, per_type, base = real(model, batches)
            worth.fill_(-1.0)
            return worth, {kind: [-1.0] for kind in per_type}, base

        search.group_worth = worthless
    if "noscale" in flags:
        multi.Group.average_ = lambda self, tensors: None
    pretrain.main(argv)


def grads(out):
    """Every process computes the gradient of its own batch; after average_grads each must hold the mean. Process 0
    works the mean out alone, from copies of the model, and writes how far off the shared result is."""
    grp = multi.Group(torch.device("cpu"))
    torch.manual_seed(0)
    cfg = get_config("tiny")
    model = ExeaLM(cfg)

    def backward(m, rank):
        x = torch.randint(0, cfg.vocab_size, (4, 33), generator=torch.Generator().manual_seed(100 + rank))
        m(x[:, :-1], MemoryContext(swarm_size=2), x[:, 1:])[1].backward()
        if rank == 1:
            m.layers[0].inp.bias.grad = None              # stands for a parameter that only some processes used
        return [None if p.grad is None else p.grad.clone() for p in m.parameters()]

    alone = [backward(copy.deepcopy(model), r) for r in range(grp.world)] if grp.lead else None
    backward(model, grp.rank)
    grp.average_grads(model)
    if grp.lead:
        worst, unused, partly = 0.0, 0, 0
        for i, p in enumerate(model.parameters()):
            parts = [g[i] for g in alone]
            if all(g is None for g in parts):
                assert p.grad is None                     # no process used it: still no gradient
                unused += 1
                continue
            partly += any(g is None for g in parts)
            want = sum(torch.zeros_like(p) if g is None else g for g in parts) / grp.world
            worst = max(worst, float((p.grad - want).abs().max()))
        json.dump(dict(processes=grp.world, worst=worst, unused=unused, partly=partly), open(out, "w"))
    grp.close()


if __name__ == "__main__":
    if sys.argv[1] == "grads":
        grads(sys.argv[2])
    else:
        cut = sys.argv.index("--")
        train(sys.argv[2:cut], sys.argv[cut + 1:])
