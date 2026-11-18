# Compute budget

What the planned runs cost, from measured throughput where we have it and stated assumptions where we don't.
Update this page when a GPU benchmark replaces an assumption.

## The main run: 0.1B parameters on 100B tokens

Training compute is about 6 × parameters × tokens = 6 × 0.1e9 × 100e9 = **6 × 10¹⁹ FLOPs**. Retrieval on adds
about half again (measured on the Mac: the 2:1 hybrid costs 1.01× a same-size Llama per token with the memory circuits
idle, 1.49× with retrieval on).

| Where | Throughput | Time for 100B tokens | Source |
|---|---|---|---|
| Mac Studio (M3 Ultra, MPS, fp32) | ~12,000 tokens/s, retrieval off; ~8,000 on | ~97 days; ~144 days | measured, `results/speed_hybrid21/` |
| One H100 or MI300X at ~300 TFLOPs sustained | ~500,000 tokens/s | ~56 GPU-hours; ~85 with retrieval | assumed, DATA.md |
| 8 of the same | | ~7 hours; ~11 hours | |

The 300 TFLOPs figure is an assumption. Attention runs as fused kernels on NVIDIA and AMD GPUs and TONN's circuits do
not yet, so real throughput could be up to 2× lower. **A one-hour benchmark on the 0.1B preset should come first**
(about $3–4 on one GPU).

## The scaling study: 2,000 GPU-hours on SDSC Expanse

The NAIRR Pilot approved 2,000 GPU-hours for ExeaLM on the SDSC Expanse AI Resource (nodes of four NVIDIA H100s,
80 GB each). The plan below is the one in the proposal. Its GPU-hours use the rates measured on one AMD MI300X in
bf16 at 0.5B ([results/funding](../results/funding/README.md), section 6): 81 effective TFLOP/s for every ExeaLM
run and 125 for every baseline. GPU-hours = 6 × parameters × tokens / (TFLOP/s × 3600 × 10¹²).

| Run | Width, layers | Training tokens | GPU-hours |
|---|---|---|---|
| ExeaLM 0.5B (`500m`) | 1280, 22 | 30B | 307 |
| Baseline 0.5B | 1280, 22 | 30B | 190 |
| ExeaLM 1B (`1b`) | 1664, 26 | 20B | 397 |
| Baseline 1B | 1664, 26 | 20B | 248 |
| ExeaLM 3B (`3b`) | 2560, 34 | 8B | 477 |
| Evaluation and short post-training of all five | | | 150 |
| Library builds, restarts and reserve | | | 231 |
| **Total** | | | **2,000** |

Two things will change these numbers, and both are measured by the first job on Expanse
([experiments/expanse](../experiments/expanse/README.md)): the rate of one H100, and the speed-up of four GPUs over
one. A whole node is charged 4 GPU-hours per hour, so a run on four GPUs that is S times as fast as on one costs
4 / S times the GPU-hours above. `experiments/throughput.py` prints the table again at the measured single-GPU rate.

If fewer hours fit: drop the 1B baseline (248 hours), then halve the tokens of the 1B run (199), then replace the 3B
run with a 2B run on the same tokens (about 160).

## What it costs to rent

RunPod on-demand list prices on 2026-11-11, per GPU-hour:

| GPU | Price | GPU-hours for the main run (retrieval on) | Cost |
|---|---|---|---|
| MI300X | $2.39 | ~85 | ~$200 |
| H100 SXM | $2.69 community, $3.99 secure | ~85 | ~$230–340 |
| A100 80GB | $1.39–1.79 | ~250 | ~$350–450 |
| RTX 4090 (8, community) | $0.34 | ~500 | ~$170, about 2.6 days on 8 |

Add 30–50% for restarts, evals and data loading. If the GPU benchmark comes in 2× slower, double these.

## The other runs

- **Data-mix pilots** (DATA.md: a 0.1B model on 20B tokens, three or four mixes): each about a fifth of the main run,
  so ~$40–80 each, ~$100–300 in all.
- **Experiments at `small` size (17M)** run on the Mac Studio: about 40,000–75,000 tokens/s, so 400M tokens in 2–3 hours.
- **Post-training** (SFT on 1–2B tokens, then preference tuning and RL): small next to pretraining, roughly 1–2% of
  its compute for SFT.

## Checkpoints

- Save every 2,000 steps and at the end (`--save_every`); keep the last three and every tenth.
- Exact resume works (`--resume auto`, PR #1), so community or spot GPUs that can be interrupted are usable.
