#!/bin/bash
#SBATCH --job-name=acdc-induction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue

# ACDC runs for the paper's headline figure: the Pareto frontier of KL divergence
# against number of edges (README, "Reproducing results").
#
# This is the README command as-is: experiments/launch_induction.py sweeps 21 thresholds
# x reset-network {0,1} x zero-ablation {0,1} = 84 runs and calls acdc/main.py for each
# one on CPU (experiments/launch_induction.py:12-45). The runs are sequential inside the
# single job, which is why the walltime is large; there is no CLI to split them, so
# --requeue is set and a preempted job simply starts the sweep again.
#
# Usage: sbatch slurm/jobs/acdc_induction.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs slurm/wandb
LOG="slurm/logs/acdc-induction-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1

module purge 2>/dev/null || true
source ~/.bashrc
conda activate python310

# experiments/launch_induction.py imports experiments.launcher, so the repo root must
# be importable; it is not installed as a top-level package.
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$SLURM_SUBMIT_DIR/slurm/wandb"

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"

# The 84 commands built by launch_induction.py do not pass --wandb-entity-name, so
# acdc/main.py:155 reads WANDB_ENTITY instead; without it the runs would go to the authors'
# entity. The group is fixed by experiments/launch_induction.py:26 to "adria-induction-3",
# and the other jobs reuse that name so all three methods end up in one group.
if [ -z "$WANDB_ENTITY" ]; then
    echo "WANDB_ENTITY is not set: add 'export WANDB_ENTITY=<your wandb user or team>' to ~/.bashrc" >&2
    exit 1
fi

python experiments/launch_induction.py

echo "Finished $(date)"
