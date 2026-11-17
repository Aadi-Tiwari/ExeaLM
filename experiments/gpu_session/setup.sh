#!/bin/bash
# One-time setup on a fresh GPU box. Run from the repo root:  bash experiments/gpu_session/setup.sh
# Needs: Python 3.10+, a PyTorch build that sees the GPU (CUDA or ROCm), and a Hugging Face token that can read the
# private dataset exea-labs/exealm-studio-data (run `huggingface-cli login` first, or export HF_TOKEN).
set -euo pipefail

python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch does not see a GPU: install the CUDA or ROCm build first"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB, torch {torch.__version__}")
PY

pip install -q -e ".[dev,data]" huggingface_hub

# the packed token files (1.2 GB) and the tokenizer
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import os, shutil
snapshot_download("exea-labs/exealm-studio-data", repo_type="dataset", allow_patterns=["tokpack/bpe32k/*"], local_dir="data")
os.makedirs("results/tokenizer/bpe32k", exist_ok=True)
shutil.copy(hf_hub_download("exea-labs/ExeaLM", "tokenizer/tokenizer.json"), "results/tokenizer/bpe32k/tokenizer.json")
need = ["train_en", "train_ja", "train_code", "train_en_big", "train_ja_big", "train_code_big", "store_wiki",
        "val_en", "val_ja_wiki", "val_ja_web", "val_code"]
missing = [n for n in need if not os.path.exists(f"data/tokpack/bpe32k/{n}/tokens.bin")]
assert not missing, f"missing token packs: {missing}"
print("data ready:", ", ".join(need))
PY

python -m pytest -q -x                                    # about a minute
python experiments/full_small.py --smoke 1 --device cuda --out /tmp/exealm_smoke > /tmp/exealm_smoke.log 2>&1 \
  && echo "smoke run ok" || { echo "smoke run FAILED: see /tmp/exealm_smoke.log"; exit 1; }
python experiments/gpu_session/make_jobs.py "$@"
echo "setup done. Next:  nohup bash experiments/gpu_session/run_all.sh > logs_gpu_session.txt 2>&1 &"
