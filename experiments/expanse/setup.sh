#!/bin/bash
# One-time setup on an Expanse login node. Run it from the repo root, inside the Python environment the jobs will use
# (a conda environment in your home directory is what SDSC suggests):
#
#     conda create -n exealm python=3.11 && conda activate exealm
#     huggingface-cli login            # or export HF_TOKEN: a token that can read exea-labs/exealm-studio-data
#     bash experiments/expanse/setup.sh
#
# Login nodes have no GPU, so nothing here touches one: the GPU checks run in first_contact.sb.
set -euo pipefail

python - <<'PY'
import sys
assert sys.version_info >= (3, 10), "Python 3.10 or newer"
PY
python -c "import torch" 2>/dev/null || pip install -q torch          # the default Linux wheel carries CUDA
pip install -q -e ".[dev,data]" huggingface_hub
python - <<'PY'
import torch
assert torch.version.cuda, f"torch {torch.__version__} is a CPU-only build: install a CUDA build before submitting jobs"
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
PY

# the packed token files (1.2 GB) and the tokenizer, into ./data and ./results (in your home directory; jobs copy
# ./data to the node's local disk when they start)
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

python -m pytest -q -x                                                 # on the login node's CPU, a minute or two
mkdir -p logs/expanse runs results/expanse
echo "setup done. Next:  export SBATCH_ACCOUNT=<your project>;  sbatch experiments/expanse/first_contact.sb"
