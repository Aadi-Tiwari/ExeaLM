"""Training on several GPUs: one process per GPU, each reading its own batches, all taking the same optimizer step.

Start a run with torchrun, which starts the processes and tells each one its rank:
    torchrun --nproc_per_node 4 -m exealm.train.pretrain --preset 500m --data mix.json ...

Every process holds the whole model. In each step every process runs its own batches forward and backward, the
gradients are averaged over the processes, and every process applies the same update, so the weights stay the same
everywhere. Process 0 leads: it alone logs, evaluates, runs the architecture search and writes checkpoints.

This is what torch's DistributedDataParallel does, written out, for three reasons:
  - Many of ExeaLM's parameters get no gradient in some runs (a retrieval circuit without a datastore, a board in a
    swarm of one). The wrapper needs a graph walk every step to cope with that. Here a parameter no process used simply
    stays without a gradient, exactly as on one GPU.
  - The search replaces circuits while the model trains. After a round, process 0 sends the new architecture, weights
    and optimizer state to the others (send_state).
  - Each TONN layer keeps a running size of every slot's output (TONNLayer.scale), measured on the batches it sees.
    Each process measures its own; they are averaged after every step (average_), so the model is the same everywhere
    and a checkpoint from process 0 holds all of it.

A swarm never spans two GPUs: each process draws whole swarms for its own batch.

Without torchrun (no RANK in the environment) there is one process and every method here does nothing or the obvious
single-process thing, so the training loop reads the same either way.

Tested with two processes on a CPU (tests/test_multi.py). Not yet run on several real GPUs.
"""
import datetime
import os

import torch
import torch.distributed as dist


class Group:
    """This process's place in a run: its rank (0 leads), how many processes there are (world) and its device."""

    def __init__(self, device):
        """device: what one process would use (pretrain.pick_device). With several processes on GPUs, process i of a
        node takes GPU i (if fewer GPUs are visible than processes, they share, which is slow but fine for a test)."""
        self.world = int(os.environ.get("WORLD_SIZE", "1")) if "RANK" in os.environ else 1
        self.rank = int(os.environ.get("RANK", "0")) if self.world > 1 else 0
        self.device = device
        if self.world == 1:
            return
        if device.type == "mps":
            raise ValueError("several processes need CUDA GPUs (or --device cpu for a test), not MPS")
        if device.type == "cuda":
            local = int(os.environ.get("LOCAL_RANK", self.rank))
            self.device = torch.device("cuda", local % torch.cuda.device_count())
            torch.cuda.set_device(self.device)
        # The others wait inside a collective call while process 0 evaluates, searches or saves. The default limit on
        # that wait is minutes, and a search round on a large model can take longer.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=datetime.timedelta(hours=6))

    @property
    def many(self):
        return self.world > 1

    @property
    def lead(self):
        return self.rank == 0

    def average_grads(self, model):
        """Replace every gradient by its mean over the processes. A parameter that got a gradient on some processes
        only counts as zero on the others; one that got none anywhere is left without (the optimizer then skips it)."""
        if not self.many:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        has = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=self.device)
        dist.all_reduce(has)
        for p, n in zip(params, has.tolist()):
            if n == 0:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad.div_(self.world)

    def average_(self, tensors):
        """Each tensor becomes its mean over the processes, in place."""
        if not self.many:
            return
        for t in tensors:
            dist.all_reduce(t)
            t.div_(self.world)

    def total(self, values):
        """A few numbers summed over the processes, as a float64 tensor (one process: the numbers themselves)."""
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        if self.many:
            dist.all_reduce(t)
        return t

    def gather(self, obj):
        """One small Python object from every process, as a list in rank order."""
        if not self.many:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def share(self, obj):
        """Process 0's small Python object, on every process."""
        if not self.many:
            return obj
        box = [obj if self.lead else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def send_state(self, model, opt=None):
        """Process 0's weights, buffers and optimizer statistics, to every process."""
        if not self.many:
            return
        with torch.no_grad():
            for t in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(t.data, src=0)
            if opt is not None:
                for p in model.parameters():
                    for name, v in sorted(opt.state.get(p, {}).items()):
                        if torch.is_tensor(v) and v.dim() > 0:
                            dist.broadcast(v, src=0)

    def in_sync(self, model):
        """Do all processes hold the same model? Compares one number per process (the sum of every weight and
        buffer), so it costs nothing next to a step. They should never differ; if they ever do, the caller re-sends
        process 0's state instead of training on with models that have drifted apart."""
        if not self.many:
            return True
        with torch.no_grad():
            s = sum(t.sum(dtype=torch.float64) for t in list(model.parameters()) + list(model.buffers()))
        both = torch.stack([s, -s])
        dist.all_reduce(both, op=dist.ReduceOp.MAX)                      # the largest sum, and minus the smallest
        return bool(both[0] == -both[1])

    def close(self):
        if self.many and dist.is_initialized():
            dist.destroy_process_group()
