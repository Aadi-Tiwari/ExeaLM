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

## 1. 13M parameters, five seeds

{{TABLE_13M}}

{{PAIRS_13M}}

- With the original starting menu, ExeaLM's median (1.511) is ahead of every baseline's, and seeds 0 to 2 beat every
  baseline run. Seed 4 drew a poor set of starting circuits (four rotary-attention circuits, no state-space or
  oscillating memory) and scored 1.567, mostly on code. That one seed pulls the mean level with the Qwen3.5-style
  model and the gated Llama, and makes ExeaLM's spread several times theirs.
- The circuit search did not repair it: seed 4 with the search on scored 1.565 (seed 3: 1.520). The search as tuned
  (every 250 steps, first half of training) replaced one circuit. Seeds 0 to 2 with the search were not run.
- Leaving the rotary-attention circuit out of the starting menu did (`"init_weights": {..., "RP": 0}`): all five
  seeds land between 1.503 and 1.515, each ahead of every Qwen3.5-style run and every gated-Llama run. Rotary
  attention duplicates what the attention layers of the hybrid already do. This menu was chosen after seeing seed 4,
  so treat it as a follow-up to confirm on new seeds, not as a pre-planned comparison.
- Seeds 0 and 1 reproduce the Mac Studio runs of 12 November (1.511 and 1.512 against 1.520 and 1.521 for the
  Qwen3.5-style model).

## 2. The normalization look-ahead (#47) costs nothing to fix

A circuit with no measured output scale was scaled by the batch it was reading, so its output depended on later tokens
and on other rows until its first training step. The fix uses a constant scale until the first measurement.

{{PAIRS_FIX}}

The fix is merged. The "leak fix on" rows in this file were run with it.

## 3. About 0.1B parameters (width 640, 14 layers, TONN at layers 0, 3, 6, 9, 12)

{{TABLE_100M}}

{{PAIRS_100M}}

- Equal steps, learning rate 1e-3 for every model. On this GPU in 32-bit precision ExeaLM takes about 1.2 times as
  long per step as Llama (1.6 times in bf16, section 6), so at equal training time the gap is smaller than at equal
  steps. The earlier equal-time result on the Mac Studio (1.455 against 1.479, 2 seeds, no gated attention) is in
  `results/README.md`, section 8.
- ExeaLM has 4% more parameters than the plain Llama (93.4M against 89.7M): the gate and QK-norm of its attention
  layers. The gated Llama row is the same-parameter control.

## 4. The relay task: the board and the weight-alter circuit (`experiments/swarm_relay.py --memory AG|WA`)

One copy of the model reads four facts, another must answer questions about them, and only the path under test
connects them. Chance is 1.6%. 0.8M parameters, 3,000 steps.

{{RELAY}}

- The board passes the facts almost every time. Weight-alter passes about half, on every seed and in every variant
  tried: 1, 2 or 4 slots per layer, writes corrected so they do not disturb each other (`exact_writes`, 49 to 51% when
  stopped at 1,750 to 2,250 steps), learned positions for the key and the value (41 to 45% at 750 to 1,000 steps),
  and edits to the MLP's input matrix. Logs of the stopped runs are in `results/weight_alter/logs/`. A 12,000-step
  run was stopped at step 3,000 at 49.5%.
- 52% is what an asker scores if it knows which four answers exist, not which question each belongs to, and rules
  out the answers it has already seen. So the likely reading is that weight changes carry the values across but the
  model has not learned to bind them to keys. This was not checked question by question.
- On text, one self-editing weight-alter slot per TONN layer costs about 0.008 bits per byte at 13M (section 1,
  2 seeds, original write rule).

## 5. One 0.1B model with the board, the library and weight-alter (`experiments/full_small.py --preset 100m --wa 1`)

The "together" recipe of `results/README.md` section 10 (relay and lookup taught with the text from the first step),
with one weight-alter slot added to every TONN layer, and a same-size Llama on the same stream of batches. One seed,
learning rate 2e-3 for both.

{{COMBINED}}

- Both skills form: relay 99.4% (0.8% with the board cut), lookup 77.6% on 1,024 unseen facts (1.6% without the
  library) and still rising. The Llama stays at chance on both.
- The relay is carried by the board alone: switching the weight-alter slots off leaves it at 99.4%, switching the
  board off leaves 0.7%.
- On Wikipedia the library's gain shrinks from 0.14 to 0.007 bits per token as the model learns the text, and its
  search finds the right article 26% of the time, as in earlier runs.
- On ordinary text ExeaLM is far ahead of the Llama here (English 1.706 against 2.046). Do not rely on the size of
  that gap: one seed, and one learning rate that earlier tests found too high for a Llama of this size (its best was
  1e-3). Section 3 is the fair text comparison.

## 6. Training speed on the MI300X (other jobs paused)

{{THROUGHPUT}}

Effective TFLOP/s is 6 x parameters x tokens per second. No compiled or fused kernels for the TONN circuits.

## What this does not show

- Short runs: 49M tokens, far from convergence.
- The Qwen3.5-style baseline is a small build of the layout, not the released models.
- The memory circuits are shown on tasks built for them. None has lowered the loss on ordinary text.
- The no-rotary-attention menu was chosen after the fact.
- Nothing above 0.1B.
