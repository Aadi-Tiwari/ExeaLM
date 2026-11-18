# ExeaLM on Expanse: H100 nodes, four GPUs at a time

How the scaling study (0.5B, 1B and 3B, from our NAIRR Pilot allocation of 2,000 GPU-hours) runs on the SDSC Expanse AI
Resource. **Nothing here has run on Expanse or on any NVIDIA GPU yet.** What has been checked, on a laptop CPU: training
with two processes (the tests in `tests/test_multi.py`), the GPU check with the CPU standing in for the GPU, and
`train.sb` end to end with Slurm's commands faked (a run stopped by its time limit, copied off the "node", resubmitted
and finished by the next job).

## The machine

From the [Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html), read on 18 November 2026:

| | |
|---|---|
| Node | 4 NVIDIA H100 (80 GB each), two 36-core Intel CPUs, 1 TB of memory, 6.4 TB of local NVMe |
| Partitions | `nairr-gpu`: whole nodes, up to 4 per job. `nairr-gpu-shared`: one node, fewer than 4 GPUs |
| GPUs | requested with their type: `--gpus=h100:4` |
| Time limit | 48 hours per job |
| Charge | GPUs × hours. A whole node costs 4 GPU-hours per hour |
| Local disk | `/scratch/$USER/job_$SLURM_JOBID`, emptied when the job ends |
| Storage that stays | home (100 GB) and Ceph through S3. The Lustre filesystem is not mounted on these nodes |
| Login | `login.expanse.sdsc.edu`, the same as the rest of Expanse |

Three things follow. A run has to use all four GPUs or it wastes three quarters of what it is charged, so training
runs as four processes (`exealm/train/multi.py`). A run longer than 48 hours has to stop and continue, and since the
local disk is emptied, its checkpoint has to be copied off the node first. And the data has to be copied onto the
node at the start of every job.

## 1. Setup, on a login node

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
conda create -n exealm python=3.11 && conda activate exealm
huggingface-cli login                       # a token that can read exea-labs/exealm-studio-data (private)
bash experiments/expanse/setup.sh           # installs, downloads 1.2 GB of token packs into ./data, runs the tests
export SBATCH_ACCOUNT=<your project>        # every sbatch below needs it
```

## 2. The first job: is it right, and how fast is it?

```bash
sbatch experiments/expanse/first_contact.sb          # about an hour on one node: 4 GPU-hours
```

It writes to `results/expanse/`:

| File | What it answers |
|---|---|
| `gpu_check.txt` | Does every circuit type, and each whole model, compute on the H100 what it computes on the CPU, in 32-bit and in bf16? Does a short run train, do two processes on two GPUs stay in sync, and is a resumed run identical to the last bit? |
| `throughput_h100.json` | Tokens per second on one H100 at 0.1B and 0.5B (the same measurement as `results/funding/throughput_mi300x.json`), at 1B and 3B, with the peak memory of each, and at 0.5B with `torch.compile` and with TF32. The job's output also shows what each run of the compute plan costs at these rates |
| `profile_circuits_*.json` | Which circuit types gain least from bf16 (the open to-do from the MI300X session) |
| `scaling_500m.json` | The real training loop at 0.5B on one GPU and on four: tokens per second, and the speed-up |

`gpu_check.py` and `throughput.py` run on any NVIDIA card, so the correctness part can be done before the allocation
starts: `python experiments/gpu_check.py`.

**Read `scaling_500m.json` before planning.** The compute plan counts GPU-hours as if every run had one GPU. On four
GPUs a run costs 4 GPU-hours per hour, so if four GPUs are 3.6 times as fast as one, every token costs 4 / 3.6 = 1.11
times what the plan budgeted. Rescale the token counts by that and by the H100 rate together.

## 3. Training runs

```bash
export RUN=pilot_500m_s0
export ARGS="--preset 500m --data experiments/expanse/mix_tokpack.json --val data/tokpack/bpe32k/val_en --seq 512 --batch 16 --steps 3000 --lr 1e-3"
sbatch experiments/expanse/train.sb
```

That is a pilot: 3,000 steps of 32,768 tokens, 98M tokens from the packs `setup.sh` downloads (English, Japanese and
code at 0.5, 0.3 and 0.2, 413M tokens in all). **The runs of the compute plan need the full data packed first:** 30
billion tokens at 0.5B would read these packs 73 times.

`train.sb` copies `./data` to the node's NVMe, continues the run from its last checkpoint if it has one, trains on
the job's four GPUs, copies every new checkpoint back to `./runs/$RUN`, stops cleanly 45 minutes before the time
limit and submits the next job, until the run has written `model.pt`. Its settings are listed at the top of the file.
A job that crashes is not resubmitted.

**What one step is.** `--batch` is the rows each GPU reads in one forward pass and `--accum` how many passes go into
one optimizer step. One step trains on batch × seq × accum × GPUs tokens (16 × 512 × 1 × 4 = 32,768 above), and
`--steps`, the warm-up and the learning-rate schedule count those steps. Every evaluation line in `log.jsonl` has the
tokens seen so far (`tokens`), the rate since the last one (`tokens_per_sec`), the peak memory of GPU 0
(`gpu_mem_gb`) and whether the four processes still hold the same model (`processes_in_sync`, which should always be
true).

**A baseline** is the same line with `--model llama` (add `--config` with `attn_gate` and `qk_norm` for a gated
Llama). **One GPU instead of four**, for a small run:
`export SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1` before `sbatch`.

**Where runs are kept.** A checkpoint holds the weights and two optimizer statistics per weight, 12 bytes per
parameter: about 6 GB at 0.5B, 12 GB at 1B and 35 GB at 3B, and a finished run holds two files of that size. The
default, `./runs` in your 100 GB home directory, is enough for the 0.5B pair. For the larger runs set
`KEEP=s3://<bucket>/runs` (and `DATA=s3://<bucket>/data` once the packed data outgrows home), which uses `s3cmd`.

**Memory at 3B.** Weights, gradients and the two optimizer statistics take 16 bytes per parameter in 32-bit: 46 GB of
an H100's 80 GB at 2.9B parameters, before any activations. That is arithmetic, not a measurement.
`throughput.py --sizes 3B` reports the real peak and halves the batch until it fits. Use `--accum` to keep the step
size.

## To check on the first login

We could not check these from outside:

- The `#SBATCH` lines for tasks and cores in both job files are our guess. SDSC's own sample is in
  `/cm/shared/examples/sdsc/ExpanseAIR/pytorch`: copy what differs (a `--mem` line, for example).
- Whether compute nodes reach the internet (Weights & Biases, Hugging Face). The job files assume they do not: the
  data is downloaded on the login node by `setup.sh`.
- The S3 endpoint and bucket for the Ceph storage, and whether `s3cmd` is configured for it.
- Whether the default `pip install torch` matches the nodes' NVIDIA driver. `first_contact.sb` prints both.

## Not done yet

- A model that does not fit one GPU. Every process holds the whole model, so this is not a way around the 80 GB.
- More than one node per run. The code does not care, but `train.sb` starts its processes on one node.
- The retrieval library searched across GPUs: each process holds and searches the whole datastore.
- Fused kernels for the scans ([#16](https://github.com/exea-labs/ExeaLM/issues/16)). `profile_circuits_*.json` from
  the first job says which circuit to start with.
