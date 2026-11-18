import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from exealm import ExeaLM, MemoryContext, count_params


path = Path(__file__).resolve().parents[1] / "experiments" / "swarm_relay.py"
spec = importlib.util.spec_from_file_location("swarm_relay", path)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


@pytest.mark.parametrize("facts", [4, 8, 16, 32, 64])
def test_relay_scores_each_fact_once_and_after_its_write(facts):
    chunk = 8
    x, y = relay.make_batch(3, facts, chunk, np.random.default_rng(0))
    assert x.shape == y.shape == (6, 2 * facts * chunk)
    for writer, asker, targets in zip(x[::2], x[1::2], y[1::2]):
        writes = writer.view(-1, chunk)[:facts, :3]
        reads = asker.view(-1, chunk)[facts:, :3]
        assert (writes[:, 0] == relay.FACT).all()
        assert (reads[:, 0] == relay.ASK).all()
        answers = dict(writes[:, 1:].tolist())
        assert len(answers) == facts
        assert all(relay.KEY0 <= key < relay.KEY0 + relay.N_KEYS for key in answers)
        assert all(relay.VAL0 <= val < relay.VAL0 + relay.N_KEYS for val in answers.values())
        assert dict(reads[:, 1:].tolist()) == answers
        assert (targets >= 0).nonzero().flatten().tolist() == [
            (facts + j) * chunk + 1 for j in range(facts)
        ]
        assert targets[targets >= 0].tolist() == reads[:, 2].tolist()
    assert (y[::2] == -100).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_sweep_matches_non_memory_slot_layouts(seed):
    shared = dict(pool="mean", board_chunk=0, d=128, layers=4, slots=8, h=16, chunk=8,
                  agents=2, seed=seed, wa_scope="others", wa_matrix="out", wa_key="hidden")
    ag = ExeaLM(relay.relay_config(Namespace(**shared, memory="AG"), matched=True))
    wa = ExeaLM(relay.relay_config(Namespace(**shared, memory="WA"), matched=True))
    assert ag.encoder is wa.encoder is None
    assert not ag.cfg.board_read_self
    assert wa.cfg.wa_scope == "others"
    assert wa.cfg.wa_key == "hidden"
    for ag_slots, wa_slots in zip(ag.specs, wa.specs):
        assert ag_slots[0]["tmpl"] == "AG"
        assert wa_slots[0]["tmpl"] == "WA"
        assert ag_slots[1:] == wa_slots[1:]
    assert count_params(wa) == pytest.approx(count_params(ag), rel=0.05)


@pytest.mark.parametrize("facts", [32, 64])
@pytest.mark.parametrize("memory", ["AG", "WA"])
@torch.no_grad()
def test_long_relay_forward_respects_the_communication_cut(facts, memory):
    chunk = 8
    args = Namespace(pool="mean", board_chunk=0, d=32, layers=1, slots=4, h=8, chunk=chunk,
                     agents=2, seed=0, memory=memory, wa_scope="others", wa_matrix="out", wa_key="hidden")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = ExeaLM(relay.relay_config(args, matched=True)).eval()
    x, y = relay.make_batch(1, facts, chunk, np.random.default_rng(args.seed))
    changed = x.clone()
    values = changed[0, 2:facts * chunk:chunk]
    values.copy_(relay.VAL0 + (values - relay.VAL0 + 1) % relay.N_KEYS)
    questions = y >= 0

    for swarm_size in (2, 1):
        context = MemoryContext(swarm_size=swarm_size)
        logits, loss = model(x, context, y)
        assert logits.shape == (2, 2 * facts * chunk, model.cfg.vocab_size)
        assert torch.isfinite(loss)
        answers = logits[questions]
        changed_answers = model(changed, context)[0][questions]
        assert answers.shape[0] == facts
        assert torch.isfinite(answers).all()
        assert torch.isfinite(changed_answers).all()
        if swarm_size == 1:
            torch.testing.assert_close(answers, changed_answers, rtol=0, atol=0)
        else:
            assert not torch.allclose(answers, changed_answers)
