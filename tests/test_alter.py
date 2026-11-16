"""The weight-alter circuit: a slot that changes a weight matrix for its own agent or for its swarm."""
import pytest
import torch

from exealm import ExeaLM, MemoryContext, get_config
from exealm.config import MENU
from exealm.model.alter import apply_edits, exact_writes, weight_delta


def model(scope="all", swarm=2, target=-1, seed=0):
    torch.manual_seed(seed)
    cfg = get_config("tiny", pinned={"WA": 1}, menu=tuple(t for t in MENU if t not in ("RG", "AG")) + ("WA",),
                     wa_scope=scope, wa_target=target, swarm_size=swarm)
    m = ExeaLM(cfg)
    with torch.no_grad():
        for lay in m.layers:
            lay.wa.to_u.weight.normal_(std=0.5)            # a fresh circuit barely writes: turn it up for the tests
    m.train()
    m(torch.randint(0, 256, (4, 32)), MemoryContext(swarm_size=swarm))     # measure the circuit scales once
    return m.eval()


def test_wa_is_opt_in_and_old_models_have_no_such_parameters():
    assert "WA" not in MENU
    plain = ExeaLM(get_config("tiny"))
    assert all(lay.wa is None for lay in plain.layers)
    assert not any(".wa." in k for k in plain.state_dict())
    assert all(any(s["tmpl"] == "WA" for s in lay.specs) for lay in model().layers)
    with pytest.raises(ValueError, match="wa_scope"):
        ExeaLM(get_config("tiny", pinned={"WA": 1}, wa_scope="everyone"))


def test_a_change_never_reaches_the_chunk_that_made_it_or_earlier():
    m = model(scope="all", swarm=1)
    x = torch.randint(0, 256, (2, 32))
    y = x.clone()
    y[:, 20] = (y[:, 20] + 1) % 256                         # inside chunk 2 (tokens 16..23)
    with torch.no_grad():
        a, b = m(x)[0], m(y)[0]
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-6)
    assert not torch.allclose(a[:, 24:], b[:, 24:], atol=1e-6)


@pytest.mark.parametrize("scope,reaches_other,reaches_self", [("others", True, False), ("self", False, True),
                                                               ("all", True, True)])
def test_scope_decides_whose_weights_change(scope, reaches_other, reaches_self):
    m = model(scope=scope)
    x = torch.randint(0, 256, (2, 32))
    seen = {}
    def record(mod, inp, out):                              # the changes emitted so far in this pass
        seen.setdefault(0, inp[1].cache["wa"])

    m.layers[0].wa.register_forward_hook(record)
    with torch.no_grad():
        m(x, MemoryContext(swarm_size=2))
    edit = next(iter(seen[0].values()))[0]
    assert edit["u"].shape[:3] == (2, 4, 1) and edit["swarm"] == 2
    # the literal weight change agent 1 runs with in the last chunk: built from agent 0's posts, its own, or both
    d = weight_delta(edit, row=1, position=31)
    from_other = torch.einsum("cd,cf->df", edit["u"][0, :3, 0], edit["v"][0, :3, 0])
    from_self = torch.einsum("cd,cf->df", edit["u"][1, :3, 0], edit["v"][1, :3, 0])
    want = (from_other if reaches_other else 0) + (from_self if reaches_self else 0)
    assert torch.allclose(d, want, atol=1e-6) and float(d.abs().max()) > 0
    # and the other agent's text changes agent 1's predictions only when its weights are in scope
    y = x.clone()
    y[0] = torch.randint(0, 256, (32,))
    with torch.no_grad():
        a, b = m(x, MemoryContext(swarm_size=2))[0][1], m(y, MemoryContext(swarm_size=2))[0][1]
    assert torch.allclose(a, b, atol=1e-6) != reaches_other


def test_the_change_is_a_real_weight_edit():
    """Adding the emitted delta to the stored matrix gives the same output as the circuit's own path."""
    m = model(scope="all")
    x = torch.randint(0, 256, (2, 32))
    got = {}
    m.mlps[0].register_forward_hook(lambda mod, inp, out: got.update(x=inp[0], edits=inp[1], out=out))
    with torch.no_grad():
        m(x, MemoryContext(swarm_size=2))
        mlp, edit = m.mlps[0], got["edits"][0]
        a, b = mlp.wi(mlp.norm(got["x"])).chunk(2, -1)
        hidden = torch.nn.functional.silu(a) * b
        assert torch.allclose(got["out"], got["x"] + mlp.wo(hidden) + apply_edits(hidden, got["edits"]), atol=1e-6)
        for row, pos in [(0, 31), (1, 17), (1, 5)]:
            w = mlp.wo.weight + weight_delta(edit, row, pos)                 # the matrix this agent runs with here
            assert torch.allclose(got["out"][row, pos], got["x"][row, pos] + hidden[row, pos] @ w.T, atol=1e-5)
        assert float(weight_delta(edit, 1, 5).abs().max()) == 0             # first chunk: nothing has been written


def test_switching_the_slot_off_removes_its_changes_and_training_reaches_it():
    m = model(scope="others")
    x = torch.randint(0, 256, (2, 32))
    y = x.clone()
    y[0] = torch.randint(0, 256, (32,))
    with torch.no_grad():
        for lay in m.layers:
            lay.mask[[k for k, s in enumerate(lay.specs) if s["tmpl"] == "WA"]] = 0
        assert torch.allclose(m(x, MemoryContext(swarm_size=2))[0][1], m(y, MemoryContext(swarm_size=2))[0][1], atol=1e-6)
        for lay in m.layers:
            lay.mask.fill_(1)
    m.train()
    tgt = torch.full((2, 32), -100)
    tgt[1, 24:] = x[1, 24:]                                 # only agent 1's last chunk is scored
    m(x, MemoryContext(swarm_size=2), tgt)[1].backward()
    wa = m.layers[0].wa
    assert all(float(p.grad.abs().sum()) > 0 for p in (wa.to_u.weight, wa.to_v.weight, wa.gate_w))
    # agent 0 got no loss of its own: its input projection learns only through the weights it changed for agent 1
    assert float(m.emb.weight.grad[x[0, :8]].abs().sum()) > 0


def test_a_later_layer_can_be_the_target_and_an_earlier_one_cannot():
    m = model(target=1)
    with torch.no_grad():
        m(torch.randint(0, 256, (2, 32)), MemoryContext(swarm_size=2))
    bad = ExeaLM(get_config("tiny", pinned={"WA": 1}, wa_target=0, swarm_size=2))
    with pytest.raises(ValueError, match="before the emitting layer"):
        bad(torch.randint(0, 256, (2, 32)), MemoryContext(swarm_size=2))


def test_every_stored_key_returns_its_own_value_after_later_writes():
    torch.manual_seed(0)
    u = torch.randn(3, 6, 2, 16)
    v = torch.nn.functional.normalize(torch.randn(3, 6, 2, 24) + 2.0, dim=-1)      # correlated keys: heavy crosstalk
    cu, cv = exact_writes(u, v)
    for upto in (1, 3, 6):
        delta = torch.einsum("begd,begf->bgdf", cu[:, :upto], cv[:, :upto])        # the matrix change so far
        back = torch.einsum("bgdf,begf->begd", delta, v[:, :upto])
        assert torch.allclose(back, u[:, :upto], atol=1e-4)
    naive = torch.einsum("begd,begf->bgdf", u, v)
    assert not torch.allclose(torch.einsum("bgdf,begf->begd", naive, v), u, atol=0.5)   # plain sums do not
    same = torch.cat([v[:, :1], v[:, :1]], 1)                                      # a repeated key stays bounded
    cu2, _ = exact_writes(u[:, :2], same)
    assert torch.isfinite(cu2).all() and float(cu2.abs().max()) < 1e3
