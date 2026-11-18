# The next GPU session: what to run, in what order, and why

Everything here runs from the repo root on one GPU with 48 GB or more (written for an AMD MI300X, 192 GB; any CUDA or
ROCm PyTorch works). Nothing in this folder has been run on a GPU yet. The scripts were smoke-tested on a CPU with
tiny models: the job runner, the 64-fact relay, bf16 and the time budget in `tokenizer_lm.py`, and the new logging in
`full_small.py`.

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
huggingface-cli login                                   # a token that can read exea-labs/exealm-studio-data (private)
bash experiments/gpu_session/setup.sh                   # checks the GPU, installs, downloads 1.2 GB of token packs,
                                                        # runs the tests and a smoke run, writes the job files
nohup bash experiments/gpu_session/run_all.sh > logs_gpu_session.txt 2>&1 &

tail -f logs_gpu_session.txt                            # one line per finished job
python experiments/gpu_session/summarize.py             # tables of what has finished so far
```

`run_all.sh` survives a closed terminal. Run it again after a crash and it continues: finished jobs are skipped.
On a node with several GPUs it pins every job to one GPU and runs that many times more jobs at once.
When it prints `ALL PHASES FINISHED`, copy `results/gpu_session/` and `logs/gpu_session/` off the box before
destroying it (the `.pt` files in `results/gpu_session/combined/` are 0.4 GB each and optional).

## The four phases

| Phase | Jobs | At a time | Rough time on an MI300X | The question |
|---|---|---|---|---|
| `eqtime` | 12 | 1 | 2 hours | At 0.1B the hybrid leads every baseline at equal steps (1.413 against 1.446 for a gated Llama). It is 1.6 times slower per step in bf16. Does it still lead when every model gets the same training time? |
| `combined` | 6 | 3 | 2 to 3 hours | The model with the board, the library and weight-alter trained in 2 of 5 runs. Where do the failed runs leave the good ones, and does either candidate fix help? |
| `long` | 6 | 3 | 3 to 4 hours | Do the 0.1B leads hold at 5 times the tokens (15,000 steps, 246M tokens)? |
| `small` | 63 | 8 | 1 to 2 hours | Issues #58 (how many facts weight edits carry) and #59 (which circuit types help the starting menu) |

The times are estimates from the last session's step times, not measurements of these queues.

To run only some phases, or in another order: `PHASES="eqtime combined" bash experiments/gpu_session/run_all.sh`.
"At a time" in the table is per GPU, on a GPU with 160 GB or more. Below that (an 80 GB H100) the defaults are 1, 1
and 3 per GPU, which is a guess from the MI300X counts and not a measurement. Set them yourself with
`COMBINED_PARALLEL`, `LONG_PARALLEL` and `SMALL_PARALLEL`: lower if a phase runs out of GPU memory, higher if
`nvidia-smi` shows room.

### 1. `eqtime`: equal training time at 0.1B

Four models (hybrid, gated Llama, Qwen3.5-style, Llama), three seeds, bf16, each trained for `--seconds 500`. On an
MI300X that is about 3,000 steps for the hybrid and about 4,800 for Llama. Each run times its own steps 10 to 50 to
decide how many steps fit, so **these run one at a time and nothing else may use the GPU**. On a slower GPU pass a
larger budget so the hybrid still gets about 3,000 steps: `bash experiments/gpu_session/setup.sh --seconds 900`.

Read it as: the hybrid's mean against each baseline's, seed by seed. The equal-steps numbers to compare with are in
`results/funding/README.md`, section 3. The hybrid row also shows whether bf16 costs it anything against 32-bit
(1.413 at 3,000 steps).

### 2. `combined`: why the full recipe fails in some runs

The four settings from the last session again (seeds 0 and 1 at learning rates 2e-3 and 1e-3; only seed 0 at 2e-3
trained), now with a log line every 100 steps: the loss of each kind of batch (text, wiki, relay, lookup), the
gradient norm and the size of the lookup datastore. Then two candidate fixes on the seed that failed at 2e-3:

- `growwhen`: the lookup datastore doubles only once the model has learned the current size (`--grow_when 1.0`).
- `longstart`: the first phase, relay and lookup only, lasts 800 steps instead of 400.

Read it as: in `logs/gpu_session/c100_*.log`, find the first step where a failed run's lookup or text loss leaves
the good run's. The guess to test is that a run which has not learned the lookup when the datastore grows keeps
getting lookup batches with a loss near 10, which swamp the text.

### 3. `long`: five times the tokens

Hybrid, gated Llama and Qwen3.5-style, two seeds, 15,000 steps in bf16, equal steps. These use the big token packs
(233M English, 93M Japanese, 87M code tokens), so no text repeats. Ten times the tokens would read the Japanese
pack 1.6 times; that needs more packed data first.

### 4. `small`: two open issues

- Relay with 4, 8, 16, 32 and 64 facts through the board and through weight edits (`--wa_key hidden`), 3 seeds.
- The 13M hybrid with one more circuit type left out of the starting menu, 10 types and the default, 3 seeds.

## Not in this session

Scaling to 0.5B and above (needs more data and a longer allocation), the equal-time comparison at 13M (done on the
Mac Studio earlier), and any post-training run (no pretrained checkpoint is worth post-training yet).
