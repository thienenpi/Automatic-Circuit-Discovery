#!/bin/bash
#SBATCH --job-name=16h-induction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# HISP / "16 heads" baseline for the induction task, one array task per metric.
# Unlike ACDC and SP, a single run yields the whole curve: the script ranks every
# head by gradient importance and then prunes them one at a time, logging the metrics
# and the edge count at each step (experiments/launch_sixteen_heads.py).
#
# Usage: sbatch slurm/jobs/16h_induction.slurm

METRICS=(kl_div nll)
WANDB_ENTITY=CHANGE_ME
WANDB_GROUP=acdc-repro

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs slurm/wandb

METRIC="${METRICS[${SLURM_ARRAY_TASK_ID:-0}]}"
if [ -z "$METRIC" ]; then
    echo "no metric for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

LOG="slurm/logs/16h-induction-${METRIC}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1

module purge 2>/dev/null || true
source ~/.bashrc
conda activate acdc

# The script imports the vendored TransformerLens copy under subnetwork_probing/, so the
# repo root has to be on PYTHONPATH.
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$SLURM_SUBMIT_DIR/slurm/wandb"

echo "Job $SLURM_JOB_ID on $(hostname); metric=$METRIC; started $(date)"
nvidia-smi || true

python experiments/launch_sixteen_heads.py \
    --task=induction --metric="$METRIC" \
    --device=cuda --reset-network=0 \
    --torch-num-threads="$SLURM_CPUS_PER_TASK" \
    --wandb-run-name="16h-induction-${METRIC}" \
    --wandb-project=acdc \
    --wandb-entity="$WANDB_ENTITY" \
    --wandb-group="$WANDB_GROUP" \
    --wandb-dir="$WANDB_DIR"

echo "Finished $(date)"
