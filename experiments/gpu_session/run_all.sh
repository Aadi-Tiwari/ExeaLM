#!/bin/bash
# Run every phase in order. From the repo root:
#     nohup bash experiments/gpu_session/run_all.sh > logs_gpu_session.txt 2>&1 &
# It survives a closed terminal, and running it again continues where it stopped (finished jobs are skipped).
# PHASES picks and orders the phases, e.g.  PHASES="eqtime combined" bash experiments/gpu_session/run_all.sh
# How many jobs run at once ON EACH GPU: COMBINED_PARALLEL, LONG_PARALLEL, SMALL_PARALLEL. The defaults are 3, 3 and 8
# on a GPU with 160 GB or more (what an MI300X ran) and 1, 1 and 3 below that (an 80 GB H100: a guess from the MI300X
# counts, not a measurement, so raise them if nvidia-smi shows room and lower them if a phase runs out of memory).
# Every job is pinned to one GPU, and a node with several GPUs runs that many times more jobs at once. The equal-time
# runs always get a GPU to themselves.
PHASES=${PHASES:-"eqtime combined long small"}
cd "$(dirname "$0")/../.."
read -r N_GPU GB < <(python -c "import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_properties(0).total_memory >> 30 if n else 0)")
[ "$N_GPU" -ge 1 ] || { echo "PyTorch does not see a GPU"; exit 1; }
GPUS=$(seq -s, 0 $((N_GPU - 1)))
if [ "$GB" -ge 160 ]; then big=3; small=8; else big=1; small=3; fi
declare -A PARALLEL=( [eqtime]=1 [combined]=${COMBINED_PARALLEL:-$big} [long]=${LONG_PARALLEL:-$big} [small]=${SMALL_PARALLEL:-$small} )
echo "== $N_GPU GPU(s) of $GB GB"
for phase in $PHASES; do
  n=$(( ${PARALLEL[$phase]} * N_GPU ))
  echo "== $(date '+%F %T') phase $phase ($n at a time, ${PARALLEL[$phase]} on each GPU)"
  python experiments/gpu_session/runner.py experiments/gpu_session/jobs/$phase.txt $n "$GPUS" || echo "== phase $phase had failures"
done
echo "== $(date '+%F %T') ALL PHASES FINISHED"
touch results/gpu_session/ALL_DONE
