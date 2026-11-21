# Muon against AdamW on the 13M hybrid (#21, 20 and 21 November 2026)

One RTX 4060 Laptop GPU (8 GB), PyTorch 2.14.1 with CUDA, Windows 11. Runs took 22 to 37 minutes. Other jobs
shared the CPU during some runs, so the Seconds column is not a speed comparison.

How to read the numbers. Bits per byte on held-out English, Japanese Wikipedia, Japanese web text and code; "Mean" is
the mean of those four; lower is better. Both optimizers train the same model, the 13M hybrid from results/funding
(`--models tonn_core` with `HYBRID_13M` from `experiments/gpu_session/make_jobs.py`), with the same data, batch
(32 x 512 tokens), schedule and steps (3,000, about 49M tokens), in 32-bit precision, weight decay 0.1. A seed fixes
the starting weights and the starting circuits, so the same seed gives both optimizers the same model to start from.

The data is not the data behind results/funding, which is private. These runs use public stand-ins built with
`experiments/leave_one_out_data.py --fetch` (FineWeb-Edu, Japanese Wikipedia, FineWeb-2 Japanese, codeparrot-clean
Python) and the same 32k tokenizer. Code here is Python only, so its bits per byte are not comparable. Compare the
two optimizers with each other, not with the numbers in results/funding.

The batch of 32 does not fit an 8 GB card at this size (it needs about 9 GB), so every run here uses `--accum 2`:
each batch runs as two halves of 16. Each step sees the same tokens. The gradient is the same except through TONN's
running slot scale, which updates once per half, and most on the first step, when only the first half is divided by
an unmeasured scale. Both optimizers ran the same way, so the comparison is fair, but the numbers may differ a
little from runs with one batch of 32.

## Results

Seed 0 at learning rates 1e-3, 2e-3 and 4e-3 for each optimizer. Both were best at 4e-3, the top of that range, so
each got one more run at 8e-3, which was worse for both. Then seed 1 at each optimizer's best rate.

| Optimizer | Learning rate | Seed 0 | Seed 1 | Mean | Seconds |
|---|---|---|---|---|---|
| adamw | 0.001 | 1.3721 |  | 1.3721 | 1360 |
| adamw | 0.002 | 1.3390 |  | 1.3390 | 1361 |
| adamw | 0.004 | 1.3209 | 1.3307 | 1.3258 | 1331 |
| adamw | 0.008 | 1.3377 |  | 1.3377 | 2220 |
| muon | 0.001 | 1.3448 |  | 1.3448 | 1498 |
| muon | 0.002 | 1.3102 |  | 1.3102 | 1497 |
| muon | 0.004 | 1.3041 | 1.3006 | 1.3024 | 1595 |
| muon | 0.008 | 1.3075 |  | 1.3075 | 1378 |

Best learning rate on seed 0: AdamW 0.004, Muon 0.004.
- Muon against AdamW at those rates: ahead by +0.0234 bits per byte (std 0.0094), better on 2 of 2 seeds; per seed 0: +0.0167, 1: +0.0301

Seed 0 also picked the rates, so seed 1 is the cleaner check. For scale, the same model's 10 seeds in
results/funding (section 1, "no rotary attention in the starting menu") have a standard deviation of 0.007 bits per
byte.

At each one's best rate, Muon had lower bits per byte than AdamW on both seeds, by 0.017 on seed 0 and 0.030 on
seed 1, and it was lower on all four kinds of text in both. That gap is about three times the seed-to-seed spread
above. It is still two seeds at one size (13M), on stand-in data and with `--accum 2`, so it does not show what
happens at 0.1B or on the real packs.

AdamW was best at 4e-3 here. results/funding trains this model with AdamW at 2e-3, which was 0.018 worse than 4e-3
on seed 0 with this data.

## Commands

```
python experiments/leave_one_out_data.py --fetch --raw data/raw --out data/tokpack/bpe32k
python experiments/tokenizer_lm.py --root data/tokpack --toks bpe32k --models tonn_core \
  --cfg '{"attn_layers": [1, 2, 4, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, "attn_gate": true, "qk_norm": true}' \
  --optimizer adamw --lr 4e-3 --seeds 0 --accum 2 --tag m13_adamw_lr4e-3_s0 --out results/muon
python experiments/muon_table.py results/muon
```

The second command once per row of the table, with that row's `--optimizer`, `--lr` and `--seeds`. Each result
file's `args` field has the exact arguments (these runs read the same packs from another folder, so
`root` there is different).
