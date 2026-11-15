"""WA: the weight-alter circuit. A slot that changes a weight matrix of the model while it reads, for itself or for
the other agents of its swarm.

Once per chunk of `cfg.board_chunk` tokens (default: `cfg.chunk`), every WA slot emits one rank-1 change to the TARGET
weight matrix, the output matrix W_o of a gated MLP (hidden width f -> stream width d):

    delta W  =  strength * u v^T
    u (d)      = U . chunk average of stream A          what to write (the output side)
    v (f)      = unit( V . chunk average of stream B )  which inputs it applies to (the input side)
    strength   = gain * sigmoid(w . chunk average of stream C + b)      the slot can stay quiet

The change takes effect from the next chunk on (never inside the chunk that produced it, so nothing sees the future)
and lasts for the rest of the pass: at a later position the target computes (W_o + sum of the changes so far) . hidden.
`cfg.wa_scope` says whose copy of the weights is changed: "self" (the agent's own), "others" (the other agents of its
swarm) or "all". Each agent of a swarm therefore runs with its own, different weights in that one matrix.
`cfg.wa_target` picks the layer whose MLP is changed: -1 (default) is the MLP right after the emitting TONN layer, or
a layer number at or after it (a change to an earlier layer could not take effect in the same pass).

Everything is differentiable, so what to write is learned from the ordinary loss: an agent learns to edit a
teammate's weights from the teammate's errors. The slot adds nothing to the residual stream itself; its whole effect
is the weight change. Switching the slot off (the search's worth measurement) removes its changes.

What this is, plainly: the changes live for one pass, like the in-batch board. They are fast weights (Schmidhuber
1992; Ba et al. 2016) written by a learned rule, a small hypernetwork (Ha et al. 2016) inside the layer. Permanent
edits to the stored parameters, across training steps, are not implemented.

WA is not in the default menu: add it with `menu` or `pinned` in the config (old checkpoints load unchanged).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAlter(nn.Module):
    """Runs every WA slot of one TONN layer. Each slot emits its own changes."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.wa_scope not in ("self", "others", "all"):
            raise ValueError("wa_scope must be 'self', 'others' or 'all'")
        if not cfg.ffn_width:
            raise ValueError("the weight-alter circuit edits an MLP: the model needs one (ffn != 0)")
        self.chunk, self.scope, self.target = cfg.board_chunk or cfg.chunk, cfg.wa_scope, cfg.wa_target
        self.to_u = nn.Linear(cfg.h, cfg.d, bias=False)
        self.to_v = nn.Linear(cfg.h, cfg.ffn_width, bias=False)
        self.gate_w = nn.Parameter(torch.zeros(cfg.h))
        self.gate_b = nn.Parameter(torch.zeros(()))
        self.log_gain = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.to_u.weight, std=0.02)                       # start close to "change nothing"

    def forward(self, S, mem, layer, slots):
        A, K, G = S                                                       # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        sw = mem.swarm_size if mem is not None else 1
        if sw < 1 or B % sw:
            sw = 1
        pool = lambda x: F.pad(x, (0, 0, 0, 0, 0, n * C - T)).view(B, n, C, g, h).float().mean(2)   # (B, n, g, h)
        on = layer.mask[torch.as_tensor(slots, device=A.device)].float()                         # 0: switched off
        strength = self.log_gain.exp() * torch.sigmoid(pool(G) @ self.gate_w.float() + self.gate_b.float()) * on
        u = self.to_u(pool(A)) * strength.unsqueeze(-1)                   # (B, n, g, d)
        v = F.normalize(self.to_v(pool(K)), dim=-1)                       # (B, n, g, f)
        here = mem.cache["layer"]
        target = here if self.target < 0 else self.target
        if target < here:
            raise ValueError(f"wa_target {target} is before the emitting layer {here}")
        mem.cache.setdefault("wa", {}).setdefault(target, []).append(
            dict(u=u, v=v, chunk=C, swarm=sw, scope=self.scope, complete=T // C))
        return A.new_zeros(B, T, g, h)


def visible(edit, T, device):
    """(sw, T, E) bool: may the reader (agent, position) use change e? E = sw * n * g, ordered author, chunk, slot."""
    B, n, g, _ = edit["u"].shape
    sw, C = edit["swarm"], edit["chunk"]
    author = torch.arange(sw, device=device).repeat_interleave(n * g)
    chunk = torch.arange(n, device=device).repeat_interleave(g).repeat(sw)
    vis = chunk.view(1, 1, -1) < (torch.arange(T, device=device) // C).view(1, T, 1)             # earlier chunks only
    same = author.view(1, 1, -1) == torch.arange(sw, device=device).view(sw, 1, 1)
    if edit["scope"] == "self":
        return vis & same
    if edit["scope"] == "others":
        return vis & ~same
    return vis.expand(sw, T, -1)


def apply_edits(hidden, edits):
    """What the changed weights add to the target's output: sum over visible changes of u (v . hidden).
    hidden (B, T, f). Returns (B, T, d)."""
    B, T, f = hidden.shape
    out = 0.0
    for e in edits:
        sw = e["swarm"]
        ns = B // sw
        u = e["u"].reshape(ns, -1, e["u"].shape[-1])                      # (ns, E, d)
        v = e["v"].reshape(ns, -1, f)                                     # (ns, E, f)
        coef = torch.einsum("astf,aef->aste", hidden.float().view(ns, sw, T, f), v)
        coef = coef * visible(e, T, hidden.device).unsqueeze(0)
        out = out + torch.einsum("aste,aed->astd", coef, u).reshape(B, T, -1)
    return out.to(hidden.dtype)


def weight_delta(edit, row, position):
    """The literal change (d, f) to the target matrix that agent `row` runs with at `position`: for inspection and
    tests. Equals what apply_edits uses."""
    B, n, g, d = edit["u"].shape
    sw = edit["swarm"]
    s, r = divmod(row, sw)
    u = edit["u"].reshape(B // sw, -1, d)[s]
    v = edit["v"].reshape(B // sw, -1, edit["v"].shape[-1])[s]
    vis = visible(edit, position + 1, u.device)[r, position].to(u.dtype)
    return torch.einsum("e,ed,ef->df", vis, u, v)
