#!/bin/bash
#SBATCH --job-name=acdc-induction-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue

# Finish an acdc_induction.slurm sweep that ran out of walltime. That job runs
# experiments/launch_induction.py, whose 84 commands are a plain sequential loop with no
# resume (launch_induction.py:51-53), so a requeue would start again at command 0.
#
# This job builds the same 84 commands through launch_induction.py itself and runs only
# the ones listed in ACDC_INDICES, so thresholds, seed and W&B run names stay identical
# to the commands the first job already executed.
#
# Usage: sbatch slurm/jobs/acdc_induction_resume.slurm
#        ACDC_INDICES=42,58,77-83 sbatch --export=ALL slurm/jobs/acdc_induction_resume.slurm
#        ACDC_DRY_RUN=1 bash slurm/jobs/acdc_induction_resume.slurm   # print, run nothing

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs slurm/wandb
LOG="slurm/logs/acdc-induction-resume-${SLURM_JOB_ID:-local}.out"
exec > "$LOG" 2>&1

module purge 2>/dev/null || true
source ~/.bashrc
[ -f slurm/env.sh ] && source slurm/env.sh
conda activate acdc

if [ -z "$WANDB_ENTITY" ]; then
    echo "WANDB_ENTITY is not set: add 'export WANDB_ENTITY=<your wandb user or team>' to slurm/env.sh" >&2
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set: wandb 0.13.11 rejects the new wandb_v1_ key format, so 'wandb login' cannot store it; export WANDB_API_KEY in slurm/env.sh instead" >&2
    exit 1
fi

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$SLURM_SUBMIT_DIR/slurm/wandb"
export ACDC_INDICES="${ACDC_INDICES:-77-83}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname); indices=$ACDC_INDICES; started $(date)"

poetry run python - <<'PY'
import os
import subprocess

import experiments.launch_induction as launch_induction


def parse(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


wanted = parse(os.environ["ACDC_INDICES"])
dry_run = bool(os.environ.get("ACDC_DRY_RUN"))
real_run = subprocess.run
index = -1
ran = []
failed = []


# launch_induction.main() builds all 84 commands before running any of them, so replacing
# subprocess.run is enough to skip the ones the first job already finished.
def gate(command, *args, **kwargs):
    global index
    index += 1
    if index not in wanted:
        return
    print(f"[{index}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    result = real_run(command, *args, **kwargs)
    (ran if result.returncode == 0 else failed).append(index)
    return result


subprocess.run = gate
launch_induction.main(testing=False, is_adria=False)

missing = sorted(wanted - set(range(index + 1)))
if missing:
    raise SystemExit(f"indices out of range 0..{index}: {missing}")
print(f"ok: {ran}")
if failed:
    raise SystemExit(f"failed: {failed}")
PY

echo "Finished $(date)"
