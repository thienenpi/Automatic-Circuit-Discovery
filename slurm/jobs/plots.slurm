#!/bin/bash
#SBATCH --job-name=acdc-plots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Turn the W&B runs of the three methods into JSON and then into figures.
# The Makefile in experiments/results/plots_data already holds one
# roc_plot_generator.py invocation per JSON file, so the "induction" target is exactly
# the set of files this reproduction needs; make_plotly_plots.py then reads that
# directory and writes experiments/results/plots/.
#
# Submit this only after the acdc / sp / 16h jobs have finished: roc_plot_generator.py
# pulls their results from the W&B API, it does not read anything local.
#
# Usage: sbatch slurm/jobs/plots.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs
LOG="slurm/logs/acdc-plots-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1

module purge 2>/dev/null || true
source ~/.bashrc
conda activate acdc

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

# roc_plot_generator.py queries the authors' entity and their historical group names by
# default; these two variables redirect it to the runs the other jobs produced.
if [ -z "$WANDB_ENTITY" ]; then
    echo "WANDB_ENTITY is not set: add 'export WANDB_ENTITY=<your wandb user or team>' to ~/.bashrc" >&2
    exit 1
fi
export ACDC_WANDB_GROUP=adria-induction-3

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"

# roc_plot_generator.py:210 skips a target whose JSON already exists, and the repo ships
# 32 of the authors' JSON files. Remove the induction ones first if you want your own runs
# summarised instead:
#   rm -f experiments/results/plots_data/*induction*.json
make -C experiments/results/plots_data induction || exit 1

python notebooks/make_plotly_plots.py

echo "Finished $(date)"
