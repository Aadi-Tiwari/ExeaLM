#!/bin/bash
# Run every phase in order. From the repo root:
#     nohup bash experiments/gpu_session/run_all.sh > logs_gpu_session.txt 2>&1 &
# It survives a closed terminal, and running it again continues where it stopped (finished jobs are skipped).
# PHASES picks and orders the phases, e.g.  PHASES="eqtime combined" bash experiments/gpu_session/run_all.sh
# The number after each name is how many jobs run at once; lower it if a phase runs out of GPU memory.
PHASES=${PHASES:-"eqtime combined long small"}
declare -A PARALLEL=( [eqtime]=1 [combined]=${COMBINED_PARALLEL:-3} [long]=${LONG_PARALLEL:-3} [small]=${SMALL_PARALLEL:-8} )
cd "$(dirname "$0")/../.."
for phase in $PHASES; do
  echo "== $(date '+%F %T') phase $phase (${PARALLEL[$phase]} at a time)"
  python experiments/gpu_session/runner.py experiments/gpu_session/jobs/$phase.txt ${PARALLEL[$phase]} || echo "== phase $phase had failures"
done
echo "== $(date '+%F %T') ALL PHASES FINISHED"
touch results/gpu_session/ALL_DONE
