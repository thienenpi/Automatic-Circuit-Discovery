#!/bin/bash
#SBATCH --job-name=sp-induction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-10

# Subnetwork Probing baseline points for the induction task. One array task per
# regularisation value: lambda_reg trades circuit size against loss, so the array is what
# produces the baseline's curve. The README leaves these values to the CLI ("using the
# CLI help"), so LAMBDAS below is a choice, not a value taken from the paper.
#
# Keep --array in sync with the length of LAMBDAS.
#
# Usage: sbatch slurm/jobs/sp_induction.slurm

LAMBDAS=(0.01 0.1 0.5 1 2 5 10 30 50 100 250)
WANDB_ENTITY=CHANGE_ME
WANDB_GROUP=acdc-repro

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs slurm/wandb

LAMBDA="${LAMBDAS[${SLURM_ARRAY_TASK_ID:-0}]}"
if [ -z "$LAMBDA" ]; then
    echo "no lambda for array index ${SLURM_ARRAY_TASK_ID}; check --array against LAMBDAS" >&2
    exit 1
fi

LOG="slurm/logs/sp-induction-lam${LAMBDA}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1

module purge 2>/dev/null || true
source ~/.bashrc
conda activate acdc

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$SLURM_SUBMIT_DIR/slurm/wandb"

echo "Job $SLURM_JOB_ID on $(hostname); lambda_reg=$LAMBDA; started $(date)"
nvidia-smi || true

# --lambda_reg with an underscore: subnetwork_probing/train.py:438 declares it that way
# (subnetwork_probing/launch_grid_fill.py:118 passes --lambda-reg, which argparse rejects).
# num-examples/seq-len are the induction values the paper's launchers use.
python subnetwork_probing/train.py \
    --task=induction --lambda_reg="$LAMBDA" --loss-type=kl_div \
    --num-examples=50 --seq-len=300 --epochs=10000 --n-loss-average-runs=20 \
    --zero-ablation=0 --reset-subject=0 \
    --device=cuda --torch-num-threads="$SLURM_CPUS_PER_TASK" \
    --wandb-name="sp-induction-${LAMBDA}" \
    --wandb-project=induction-sp-replicate \
    --wandb-entity="$WANDB_ENTITY" \
    --wandb-group="$WANDB_GROUP" \
    --wandb-dir="$WANDB_DIR"

echo "Finished $(date)"
