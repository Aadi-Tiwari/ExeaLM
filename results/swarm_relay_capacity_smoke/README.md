# Relay capacity smoke test (#58)

**Setup.** Two agents, eight swarms per batch, seeds 0 and 1, 20 optimizer steps, and four held-out batches generated
with seed 10000. Each fact and question uses one eight-token chunk. Four facts take 64 tokens per agent, and 64 facts
take 1,024. Both endpoints were run on an Apple M2 Max MacBook Pro (32 GB), with PyTorch 2.15.0 and NumPy 2.5.5.

The four-layer TONN has width 128, eight slots per layer, and slot width 16. The board (814,320 parameters) and
hidden-key weight-alter (844,216 parameters, 3.7% more) each get one cross-agent slot per layer. The rest of the slot
layout is matched within seed. WA edits the same layer's MLP output matrix. Both arms receive the same batches within
each fact-count/seed pair. Evaluation uses the same held-out batches twice, first with communication enabled and then
with each agent in a swarm of size one.

- [CPU scores and per-run JSON](cpu/README.md)
- [MPS scores and per-run JSON](mps/README.md)

Each run returned one shared and one cut score per question position, with finite train and held-out losses. At
20 steps, accuracy is close to the 1.56% chance rate. These runs verify execution and scoring at the two context
lengths. Capacity measurements require the full training schedule.

**Timing.** Mean seconds for 20 optimizer steps over seeds 0 and 1. The table excludes the final evaluation.

| Facts per holder | Circuit | CPU (2 threads) | MPS |
|---:|---|---:|---:|
| 4 | AG | 0.80 | 1.40 |
| 4 | WA | 1.05 | 7.15 |
| 64 | AG | 9.75 | 3.35 |
| 64 | WA | 23.85 | 96.90 |

For the 20-run sweep below, estimate **260–330 minutes on CPU** and **850–1,050 minutes on MPS** if one device
runs both arms. These are rough projections. The endpoints are measured, the middle sizes are interpolated
linearly in fact count and in squared fact count, and evaluation time is scaled to 16 batches every 250 steps.
The projections include startup effects present in the 20-step measurements and assume the same batch size.

**Reproduce the smoke test.** From the repository root with the project environment active:

```bash
for device in cpu mps; do
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python experiments/swarm_relay.py \
    --sweep --fact_counts 4,64 --seeds 0,1 --steps 20 --swarms 8 \
    --eval_every 20 --eval_batches 4 --device "$device" \
    --out "results/swarm_relay_capacity_smoke/$device"
done
```

**Full CPU sweep command** (20 runs, 3,000 steps per run):

```bash
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python experiments/swarm_relay.py \
  --sweep --fact_counts 4,8,16,32,64 --seeds 0,1 --steps 3000 --swarms 8 \
  --eval_every 250 --eval_batches 16 --device cpu --out results/swarm_relay_capacity
```
