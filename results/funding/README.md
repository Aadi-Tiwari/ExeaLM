# Same-size comparisons, the three memory circuits and a combined 0.1B model (15 and 16 November 2026)

One session on one AMD MI300X (PyTorch 2.14, ROCm), about three hours. This file is written by
`python experiments/funding_report.py` from the result files in this folder, `results/weight_alter*/` and
`results/combined/`. Edit `README.in.md`, not this file.

How to read the numbers. Bits per byte on held-out English, Japanese Wikipedia, Japanese web text and code; "Mean" is
the mean of those four; lower is better. Every model in a table has the same width, depth, 32k tokenizer, data (50%
English, 30% Japanese, 20% code), batch (32 x 512 tokens), schedule and steps (3,000, about 49M tokens), in 32-bit
precision. A seed fixes the starting weights and, for ExeaLM, the starting circuits. The Qwen3.5-style model is our
own build of that layout (Gated DeltaNet and gated attention layers, about 3 to 1, DeltaNet in plain PyTorch), not
the released Qwen3.5 weights. Runs shared the GPU, so their wall-clock seconds are not comparable; speed is in
section 6. Commands: `experiments/tokenizer_lm.py` (the `args` field of each file holds the exact arguments).

## 1. 13M parameters, five seeds (0 to 4), then five more (5 to 9) for some rows

| Model | Parameters | Seeds | English | Japanese wiki | Japanese web | Code | Mean | Median | Min to max | Per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| ExeaLM hybrid | 13.36M | 5 | 1.440 | 1.511 | 1.394 | 1.748 | **1.523** | 1.511 | 1.505 to 1.567 | 1.511, 1.511, 1.505, 1.521, 1.567 |
| ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) | 13.36M | 10 | 1.437 | 1.503 | 1.390 | 1.713 | **1.511** | 1.511 | 1.503 to 1.527 | 1.503, 1.509, 1.514, 1.515, 1.514, 1.505, 1.504, 1.507, 1.527, 1.512 |
| ExeaLM hybrid with the leak fix (#47) | 13.36M | 8 | 1.437 | 1.507 | 1.393 | 1.725 | **1.516** | 1.511 | 1.501 to 1.544 | 1.517, 1.509, 1.501, 1.524, 1.544, 1.508, 1.512, 1.510 |
| ExeaLM hybrid, circuit search on (leak fix on) | 13.36M | 2 | 1.443 | 1.520 | 1.395 | 1.812 | **1.542** | 1.542 | 1.520 to 1.565 | 1.520, 1.565 |
| ExeaLM hybrid + one weight-alter slot per TONN layer | 13.42M | 2 | 1.439 | 1.510 | 1.394 | 1.735 | **1.520** | 1.520 | 1.519 to 1.520 | 1.520, 1.519 |
| ExeaLM hybrid + one weight-alter slot per TONN layer, key from the edited layer (leak fix on) | 13.42M | 3 | 1.437 | 1.509 | 1.394 | 1.718 | **1.514** | 1.517 | 1.504 to 1.521 | 1.521, 1.517, 1.504 |
| ExeaLM hybrid + one weight-alter slot, key from the edited layer, input matrix (leak fix on) | 13.47M | 3 | 1.436 | 1.506 | 1.393 | 1.712 | **1.512** | 1.511 | 1.505 to 1.519 | 1.519, 1.511, 1.505 |
| Qwen3.5-style | 13.51M | 10 | 1.445 | 1.513 | 1.400 | 1.731 | **1.522** | 1.522 | 1.517 to 1.529 | 1.519, 1.521, 1.529, 1.517, 1.522, 1.520, 1.522, 1.524, 1.527, 1.522 |
| Llama + gated attention + QK-norm | 13.49M | 10 | 1.440 | 1.514 | 1.400 | 1.733 | **1.522** | 1.521 | 1.515 to 1.528 | 1.525, 1.521, 1.528, 1.518, 1.520, 1.523, 1.515, 1.521, 1.521, 1.523 |
| Llama | 13.1M | 5 | 1.466 | 1.538 | 1.418 | 1.776 | **1.550** | 1.548 | 1.544 to 1.559 | 1.544, 1.559, 1.551, 1.546, 1.548 |

- ExeaLM hybrid against Qwen3.5-style: -0.001 bits per byte on average (positive: ExeaLM is better), better on 3 of 5 seeds (per seed +0.008, +0.010, +0.024, -0.004, -0.045)
- ExeaLM hybrid against Llama + gated attention + QK-norm: -0.000 bits per byte on average (positive: ExeaLM is better), better on 3 of 5 seeds (per seed +0.014, +0.010, +0.023, -0.003, -0.047)
- ExeaLM hybrid against Llama: +0.027 bits per byte on average (positive: ExeaLM is better), better on 4 of 5 seeds (per seed +0.034, +0.048, +0.046, +0.025, -0.019)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Qwen3.5-style: +0.011 bits per byte on average (positive: ExeaLM is better), better on 10 of 10 seeds (per seed +0.016, +0.011, +0.015, +0.002, +0.008, +0.014, +0.018, +0.016, +0.000, +0.010)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Llama + gated attention + QK-norm: +0.011 bits per byte on average (positive: ExeaLM is better), better on 9 of 10 seeds (per seed +0.022, +0.012, +0.014, +0.003, +0.006, +0.017, +0.011, +0.014, -0.006, +0.011)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Llama: +0.039 bits per byte on average (positive: ExeaLM is better), better on 5 of 5 seeds (per seed +0.042, +0.050, +0.037, +0.031, +0.034)

- With the original starting menu, ExeaLM's median (1.511) is ahead of every baseline's, and seeds 0 to 2 beat every
  baseline run. Seed 4 drew a poor set of starting circuits (four rotary-attention circuits, no state-space or
  oscillating memory) and scored 1.567, mostly on code. That one seed pulls the mean level with the Qwen3.5-style
  model and the gated Llama, and makes ExeaLM's spread several times theirs.
- The circuit search did not repair it: seed 4 with the search on scored 1.565 (seed 3: 1.520). The search as tuned
  (every 250 steps, first half of training) replaced one circuit. Seeds 0 to 2 with the search were not run.
- Leaving the rotary-attention circuit out of the starting menu did (`"init_weights": {..., "RP": 0}`): all five
  seeds land between 1.503 and 1.515, each ahead of every Qwen3.5-style run and every gated-Llama run. Rotary
  attention duplicates what the attention layers of the hybrid already do. This menu was chosen after seeing seed 4,
  so it was then run on seeds 5 to 9, which it had not seen: 1.505, 1.504, 1.507, 1.527, 1.512 (mean 1.511), against
  1.523 for the Qwen3.5-style model and 1.521 for the gated Llama on the same seeds. It is ahead of the gated Llama on
  4 of those 5 and level with the Qwen3.5-style model on seed 8 (1.5269 against 1.5270). The original menu on seeds
  5 to 9 (leak fix on) averages 1.520, with another weak seed (1.544).
- Seeds 0 and 1 reproduce the Mac Studio runs of 12 November (1.511 and 1.512 against 1.520 and 1.521 for the
  Qwen3.5-style model).

## 2. The normalization look-ahead (#47) costs nothing to fix

A circuit with no measured output scale was scaled by the batch it was reading, so its output depended on later tokens
and on other rows until its first training step. The fix uses a constant scale until the first measurement.

- ExeaLM hybrid with the leak fix (#47) against ExeaLM hybrid: +0.000 bits per byte on average (positive: the fix is better), better on 2 of 3 seeds (per seed -0.006, +0.002, +0.004)

The fix is merged. The "leak fix on" rows in this file were run with it.

## 3. About 0.1B parameters (width 640, 14 layers, TONN at layers 0, 3, 6, 9, 12)

| Model | Parameters | Seeds | English | Japanese wiki | Japanese web | Code | Mean | Median | Min to max | Per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) | 93.43M | 3 | 1.347 | 1.413 | 1.311 | 1.583 | **1.413** | 1.415 | 1.410 to 1.415 | 1.415, 1.415, 1.410 |
| ExeaLM hybrid (leak fix on) | 93.43M | 3 | 1.353 | 1.423 | 1.319 | 1.611 | **1.427** | 1.419 | 1.419 to 1.441 | 1.441, 1.419, 1.419 |
| Qwen3.5-style | 95.67M | 3 | 1.369 | 1.449 | 1.342 | 1.669 | **1.457** | 1.457 | 1.453 to 1.461 | 1.461, 1.453, 1.457 |
| Llama + gated attention + QK-norm | 95.47M | 3 | 1.364 | 1.440 | 1.333 | 1.646 | **1.446** | 1.447 | 1.444 to 1.447 | 1.444, 1.447, 1.447 |
| Llama | 89.73M | 3 | 1.418 | 1.496 | 1.378 | 1.743 | **1.509** | 1.509 | 1.507 to 1.510 | 1.507, 1.510, 1.509 |

- ExeaLM hybrid (leak fix on) against Qwen3.5-style: +0.031 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.019, +0.034, +0.038)
- ExeaLM hybrid (leak fix on) against Llama + gated attention + QK-norm: +0.019 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.003, +0.027, +0.028)
- ExeaLM hybrid (leak fix on) against Llama: +0.082 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.065, +0.091, +0.090)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Qwen3.5-style: +0.044 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.045, +0.039, +0.048)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Llama + gated attention + QK-norm: +0.033 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.029, +0.032, +0.037)
- ExeaLM hybrid, no rotary attention in the starting menu (leak fix on) against Llama: +0.095 bits per byte on average (positive: ExeaLM is better), better on 3 of 3 seeds (per seed +0.091, +0.096, +0.099)

- Equal steps, learning rate 1e-3 for every model. On this GPU in 32-bit precision ExeaLM takes about 1.2 times as
  long per step as Llama (1.6 times in bf16, section 6), so at equal training time the gap is smaller than at equal
  steps. The earlier equal-time result on the Mac Studio (1.455 against 1.479, 2 seeds, no gated attention) is in
  `results/README.md`, section 8.
- ExeaLM has 4% more parameters than the plain Llama (93.4M against 89.7M): the gate and QK-norm of its attention
  layers. The gated Llama row is the same-parameter control.

## 4. The relay task: the board and the weight-alter circuit (`experiments/swarm_relay.py --memory AG|WA`)

One copy of the model reads four facts, another must answer questions about them, and only the path under test
connects them. Chance is 1.6%. 0.8M parameters, 3,000 steps.

| Path | Run | Facts passed | With the path cut | By question, in the order asked |
|---|---|---|---|---|
| Board | seed 0 (from its log) | 99.9% | 2.1% |  |
| Board | seed 1 | 99.8% | 1.3% |  |
| Board | seed 2 | 99.8% | 1.5% |  |
| Weight-alter | seed 0 (from its log) | 51.1% | 1.7% |  |
| Weight-alter | seed 1 | 53.9% | 1.5% |  |
| Weight-alter | seed 2 | 53.1% | 1.6% |  |
| Weight-alter, 2 slots per layer | seed 0 | 53.1% | 1.5% |  |
| Weight-alter, 4 slots per layer | seed 0 | 53.0% | 1.5% |  |
| Weight-alter, edits to the MLP's input matrix | seed 0 | 49.5% | 1.9% |  |
| Weight-alter, edits to the MLP's input matrix | seed 1 | 45.4% | 1.6% |  |
| Weight-alter, edits to the MLP's input matrix | seed 2 | 48.4% | 1.3% |  |
| Weight-alter, key from the edited layer | seed 0 | 99.7% | 1.4% | 99%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer | seed 1 | 100.0% | 1.6% | 100%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer | seed 2 | 100.0% | 2.1% | 100%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer, 8 facts | seed 0 | 100.0% | 1.4% | 100%, 100%, 100%, 100%, 100%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer, edits to the MLP's input matrix | seed 0 | 100.0% | 1.6% | 100%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer, edits to the MLP's input matrix | seed 1 | 100.0% | 1.7% | 100%, 100%, 100%, 100% |
| Weight-alter, key from the edited layer, edits to the MLP's input matrix | seed 2 | 100.0% | 1.9% | 100%, 100%, 100%, 100% |
| Weight-alter, rerun scored by question | seed 0 | 49.5% | 1.8% | 25%, 34%, 49%, 90% |
| Weight-alter, rerun scored by question | seed 1 | 51.2% | 1.4% | 27%, 35%, 49%, 93% |
| Weight-alter, rerun scored by question | seed 2 | 52.1% | 1.7% | 30%, 36%, 50%, 93% |

- The board passes the facts almost every time. With a learned key (the first version: both vectors of an edit are
  learned maps of the writing slot's input), weight-alter passes about half, on every seed and in every variant tried:
  1, 2 or 4 slots per layer, writes corrected so they do not disturb each other (`exact_writes`, 49 to 51% when
  stopped at 1,750 to 2,250 steps), learned positions for the key and the value (41 to 45% at 750 to 1,000 steps),
  and edits to the MLP's input matrix. Logs of the stopped runs are in `results/weight_alter/logs/`.
- 52% is what an asker scores if it knows which four answers exist, not which question each belongs to, and rules
  out the answers it has already seen: 25%, 33%, 50% and 100% on the four questions in turn. The reruns scored by
  question show that pattern (25 to 30%, 34 to 36%, 49 to 50%, 90 to 93%). The edits carried the values across; the
  model did not learn to bind them to keys.
- With the key taken from the edited layer (`wa_key="hidden"`: the key of an edit is what the edited matrix itself
  receives at the picked positions of the writer's chunk, the key ROME uses), all six runs pass 99% at step 250 and
  end at 99.7 to 100%, 1.4 to 2.1% with the path cut. With eight facts: 100% (one seed).
- On text, one self-editing weight-alter slot per TONN layer costs about 0.008 bits per byte at 13M with the
  learned key (section 1, 2 seeds), 0.005 with the key from the edited layer and 0.003 with that key and edits to
  the input matrix (3 seeds each, against the same seeds of the hybrid with the leak fix).

## 5. One 0.1B model with the board, the library and weight-alter (`experiments/full_small.py --preset 100m --wa 1`)

The "together" recipe of `results/README.md` section 10 (relay and lookup taught with the text from the first step),
with one weight-alter slot added to every TONN layer, and a same-size Llama on the same stream of batches. Rows without a step number are at step 3,000.

| Model | Step | English | Mean of 4 sets | Relay (shared / cut) | Relay, board only | Relay, weight-alter only | Lookup (attached / removed) | Wikipedia gain from the library | Search hits |
|---|---|---|---|---|---|---|---|---|---|
| ExeaLM, all three circuits (94.7M), seed 0, lr 2e-3 | 1,000 | 2.202 | 2.664 | 98.8% / 1.1% | 98.8% | 0.7% | 41.1% / 1.0% | 0.1395 | 26.1% |
| ExeaLM, all three circuits (94.7M), seed 0, lr 2e-3 | 2,000 | 1.932 | 2.406 | 99.1% / 0.8% | 99.1% | 0.7% | 70.5% / 1.7% | 0.016 | 25.8% |
| ExeaLM, all three circuits (94.7M), seed 0, lr 2e-3 | 3,000 | 1.706 | 2.063 | 99.4% / 0.8% | 99.4% | 0.7% | 77.6% / 1.6% | 0.0066 | 25.9% |
| ExeaLM, weight-alter key from the edited layer, seed 0, lr 2e-3 | 3,000 | 1.726 | 2.092 | 100.0% / 0.1% | 0.0% | 100.0% | 75.0% / 1.5% | 0.0174 | 25.9% |
| ExeaLM, all three circuits, seed 1, lr 2e-3 (did not train) | 3,000 | 4.106 | 4.846 | 1.3% / 1.3% | 1.3% | 1.3% | 0.0% / 0.0% | 0.2722 | 25.9% |
| ExeaLM, all three circuits, seed 0, lr 1e-3 | 3,000 | 3.251 | 3.740 | 98.9% / 1.0% | 98.9% | 1.4% | 0.0% / 0.0% | -0.0717 | 25.9% |
| ExeaLM, all three circuits, seed 1, lr 1e-3 | 3,000 | 3.484 | 3.952 | 6.7% / 1.4% | 1.4% | 6.5% | 0.0% / 0.0% | 0.163 | 25.9% |
| Llama (89.7M), seed 0, lr 2e-3 | 1,000 | 2.440 | 2.914 | 1.3% / 1.3% |  |  | 1.2% |  |  |
| Llama (89.7M), seed 0, lr 2e-3 | 2,000 | 2.211 | 2.734 | 1.3% / 1.3% |  |  | 1.6% |  |  |
| Llama (89.7M), seed 0, lr 2e-3 | 3,000 | 2.046 | 2.501 | 1.4% / 1.4% |  |  | 1.5% |  |  |
| Llama, seed 1, lr 2e-3 | 3,000 | 2.065 | 2.504 | 1.9% / 1.9% |  |  | 1.2% |  |  |
| Llama, seed 0, lr 1e-3 | 3,000 | 2.013 | 2.453 | 1.8% / 1.8% |  |  | 1.5% |  |  |
| Llama, seed 1, lr 1e-3 | 3,000 | 2.000 | 2.450 | 1.9% / 1.9% |  |  | 1.5% |  |  |

- Seed 0 at 2e-3: both skills form. Relay 99.4% (0.8% with the board cut), lookup 77.6% on 1,024 unseen facts (1.6%
  without the library). The relay is carried by the board alone: with the weight-alter slots off it stays at 99.4%,
  with the board off it is 0.7%. On Wikipedia the library's gain shrinks from 0.14 to 0.007 bits per token as the
  model learns the text, and its search finds the right article 26% of the time, as in earlier runs.
- With the weight-alter key taken from the edited layer (seed 0, 2e-3) the relay is 100% and it is carried by the
  weight changes alone: 100% with the board off, 0.0% with the weight-alter slots off. Lookup 75.0%, English 1.726
  (1.706 with the learned key). So given both paths, the model uses weight edits and leaves the board unused.
- **The recipe is not stable at this size.** Seed 1 at 2e-3 never trained (English 4.11 bits per byte, both tasks at
  chance). At 1e-3, seed 0 learned the relay (98.9%) but not the lookup (0.0%) and its text is poor (English 3.25);
  seed 1 learned neither (English 3.48). Two of five ExeaLM runs trained. A guess, not checked: the runs that never
  learn the lookup keep getting lookup batches with a loss near 10, which swamp the text gradient.
- The Llama's learning rate does not explain the text gap to the good ExeaLM runs: English 2.046 and 2.065 at 2e-3,
  2.013 and 2.000 at 1e-3, against 1.706 and 1.726. The Llama stays at chance on both tasks in all four runs.
  Section 3 is still the fair text comparison: no tasks, three seeds, every run trained.

## 6. Training speed on the MI300X (other jobs paused)

| Size | Model | Parameters | Precision | Seconds per step | Tokens per second | Effective TFLOP/s |
|---|---|---|---|---|---|---|
| 0.1B | ExeaLM hybrid | 93.4M | bf16 | 0.1633 | 100,310 | 56.2 |
| 0.1B | ExeaLM hybrid | 93.4M | fp32 | 0.3061 | 53,525 | 30.0 |
| 0.1B | Llama | 89.7M | bf16 | 0.1037 | 157,982 | 85.1 |
| 0.1B | Llama | 89.7M | fp32 | 0.2576 | 63,614 | 34.2 |
| 0.5B | ExeaLM hybrid | 497.1M | bf16 | 0.3016 | 27,158 | 81.0 |
| 0.5B | ExeaLM hybrid | 497.1M | fp32 | 0.6403 | 12,794 | 38.2 |
| 0.5B | Llama | 474.1M | bf16 | 0.1867 | 43,883 | 124.8 |
| 0.5B | Llama | 474.1M | fp32 | 0.5364 | 15,273 | 43.4 |

Effective TFLOP/s is 6 x parameters x tokens per second. No compiled or fused kernels for the TONN circuits.

## What this does not show

- Short runs: 49M tokens, far from convergence.
- The Qwen3.5-style baseline is a small build of the layout, not the released models.
- The memory circuits are shown on tasks built for them. None has lowered the loss on ordinary text.
- The no-rotary-attention menu was chosen after the fact; it then held on five new seeds at 13M and on three at 0.1B.
- The combined 0.1B recipe trained in two of five ExeaLM runs.
- On text a self-editing weight-alter slot costs 0.003 to 0.005 bits per byte at 13M with the new key (3 seeds each).
- Nothing above 0.1B.
