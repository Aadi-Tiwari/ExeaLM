# Relay capacity: board vs hidden-key weight-alter

**Task.** One agent reads 4, 8, 16, 32, or 64 random key-value facts. A second agent answers one question per fact,
in shuffled order, after the writer has seen all the facts. Each fact and question occupies one eight-token chunk.
Chance accuracy is 1.56%.

**Setup.** Four-layer TONN, width 128, eight slots per layer, slot width 16. The board (814,320 parameters) and
weight-alter (844,216 parameters) each use one cross-agent slot per layer and share the other slot assignments
within each seed. WA keys come from the hidden activation of the same layer's MLP, and the edit targets its output
matrix. Both arms train from scratch for 3,000 AdamW steps on the same batches within each fact-count/seed pair
(seeds 0 and 1, eight two-agent swarms per batch, peak learning rate 0.002). Every 250 steps both arms see the same
16 held-out batches (seed 10000), with communication shared and with swarm size set to one for the cut.
The sweep ran on an Apple M2 Max CPU with two PyTorch threads, PyTorch 2.15.0, and NumPy 2.5.5.

| Facts | Tokens per agent | AG accuracy, shared / cut | WA accuracy, shared / cut | AG bits, shared / cut | WA bits, shared / cut | Minutes per run, AG / WA |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 98.73% / 1.66% | 99.80% / 1.85% | 0.162 / 12.236 | 0.052 / 20.130 | 2.1 / 2.6 |
| 8 | 128 | 99.12% / 1.31% | 100.00% / 1.31% | 0.087 / 15.416 | 0.000 / 19.203 | 3.3 / 4.6 |
| 16 | 256 | 99.61% / 1.74% | 100.00% / 1.32% | 0.091 / 18.662 | 0.000 / 15.102 | 5.9 / 9.3 |
| 32 | 512 | 87.19% / 1.70% | 100.00% / 1.66% | 0.570 / 25.349 | 0.000 / 8.090 | 12.3 / 22.8 |
| 64 | 1,024 | 1.61% / 1.61% | 100.00% / 1.49% | 6.001 / 6.001 | 0.000 / 12.561 | 26.2 / 59.9 |

Rows average the two seeds at step 3,000. Held-out cross-entropy is in bits per answer token. Wall times include
evaluation, and the first AG run overlapped with local tests. The 20 runs totaled approximately 298 minutes.

**What it shows.** WA transfers all 64 facts on both seeds, with 1.60% and 1.39% accuracy when the path is cut.
AG reaches 81.84% and 92.55% at 32 facts and stays near chance throughout training at 64 facts. The experiment
establishes that hidden-key WA can carry at least 64 facts in this synthetic relay. Sixty-four exhausts the task's
key vocabulary. Both circuits' learning curves and final scores are tied to the tested model and step budget.

**Reproduce.** From the repository root with the project environment active:

```bash
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python experiments/swarm_relay.py \
  --sweep --fact_counts 4,8,16,32,64 --seeds 0,1 --steps 3000 --swarms 8 \
  --eval_every 250 --eval_batches 16 --device cpu --out results/swarm_relay_capacity
```

### Per-run results

Train loss is the last training batch's cross-entropy in nats.

| Facts per holder | Seed | Memory | Tokens per agent | Parameters | Step | Train loss | Shared accuracy | Cut accuracy | Shared bits | Cut bits | JSON |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0 | AG | 64 | 814,320 | 3,000 | 0.0011 | 98.05% | 1.17% | 0.295 | 15.702 | [JSON](relay_seed0.json) |
| 4 | 0 | WA | 64 | 844,216 | 3,000 | 0.0000 | 99.61% | 0.98% | 0.104 | 20.782 | [JSON](relay_wa_hkey_seed0.json) |
| 4 | 1 | AG | 64 | 814,320 | 3,000 | 0.0032 | 99.41% | 2.15% | 0.030 | 8.771 | [JSON](relay_seed1.json) |
| 4 | 1 | WA | 64 | 844,216 | 3,000 | 0.0000 | 100.00% | 2.73% | 0.001 | 19.477 | [JSON](relay_wa_hkey_seed1.json) |
| 8 | 0 | AG | 128 | 814,320 | 3,000 | 0.0075 | 98.73% | 1.17% | 0.098 | 14.276 | [JSON](relay_a2_f8_seed0.json) |
| 8 | 0 | WA | 128 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.56% | 0.000 | 17.024 | [JSON](relay_a2_f8_wa_hkey_seed0.json) |
| 8 | 1 | AG | 128 | 814,320 | 3,000 | 0.0065 | 99.51% | 1.46% | 0.076 | 16.556 | [JSON](relay_a2_f8_seed1.json) |
| 8 | 1 | WA | 128 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.07% | 0.000 | 21.381 | [JSON](relay_a2_f8_wa_hkey_seed1.json) |
| 16 | 0 | AG | 256 | 814,320 | 3,000 | 0.0006 | 99.76% | 1.66% | 0.069 | 19.099 | [JSON](relay_a2_f16_seed0.json) |
| 16 | 0 | WA | 256 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.42% | 0.000 | 8.859 | [JSON](relay_a2_f16_wa_hkey_seed0.json) |
| 16 | 1 | AG | 256 | 814,320 | 3,000 | 0.1256 | 99.46% | 1.81% | 0.114 | 18.225 | [JSON](relay_a2_f16_seed1.json) |
| 16 | 1 | WA | 256 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.22% | 0.000 | 21.344 | [JSON](relay_a2_f16_wa_hkey_seed1.json) |
| 32 | 0 | AG | 512 | 814,320 | 3,000 | 0.6301 | 81.84% | 1.76% | 0.757 | 29.734 | [JSON](relay_a2_f32_seed0.json) |
| 32 | 0 | WA | 512 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.64% | 0.000 | 8.865 | [JSON](relay_a2_f32_wa_hkey_seed0.json) |
| 32 | 1 | AG | 512 | 814,320 | 3,000 | 0.3300 | 92.55% | 1.64% | 0.383 | 20.963 | [JSON](relay_a2_f32_seed1.json) |
| 32 | 1 | WA | 512 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.68% | 0.000 | 7.314 | [JSON](relay_a2_f32_wa_hkey_seed1.json) |
| 64 | 0 | AG | 1,024 | 814,320 | 3,000 | 4.1597 | 1.68% | 1.68% | 6.001 | 6.001 | [JSON](relay_a2_f64_seed0.json) |
| 64 | 0 | WA | 1,024 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.60% | 0.000 | 8.027 | [JSON](relay_a2_f64_wa_hkey_seed0.json) |
| 64 | 1 | AG | 1,024 | 814,320 | 3,000 | 4.1581 | 1.55% | 1.55% | 6.000 | 6.000 | [JSON](relay_a2_f64_seed1.json) |
| 64 | 1 | WA | 1,024 | 844,216 | 3,000 | 0.0000 | 100.00% | 1.39% | 0.000 | 17.094 | [JSON](relay_a2_f64_wa_hkey_seed1.json) |
