#!/usr/bin/env bash
#
# Submit the reproduction steps from the README to SLURM, one job per step.
#
#   bash slurm/submit_jobs.sh test    # 1 quick ACDC run, wandb offline (smoke test)
#   bash slurm/submit_jobs.sh acdc    # python experiments/launch_induction.py
#                                     #   -> the KL vs #edges Pareto frontier (84 runs, CPU, sequential)
#   bash slurm/submit_jobs.sh sp      # python subnetwork_probing/train.py, one job per LAMBDAS value
#   bash slurm/submit_jobs.sh 16h     # python experiments/launch_sixteen_heads.py, one job per metric
#   bash slurm/submit_jobs.sh plots   # roc_plot_generator.py via the repo Makefile, then the plots
#
# Run acdc/sp/16h first (they can run at the same time), then plots.
#
set -euo pipefail

# ------------------------------- edit these -------------------------------
CONDA_ENV=acdc
# Partition names differ per cluster: check `sinfo -s` (* marks the default) and
# `sinfo -O partition,gres` for the ones with GPUs. Leave empty to use the default.
PARTITION_CPU=
PARTITION_GPU=
GPUS=gpu:1
CPUS=4
MEM=32G
TIME_ACDC=72:00:00           # launch_induction.py runs its 84 configurations one after another
TIME=12:00:00
WANDB_ENTITY=CHANGE_ME
WANDB_PROJECT=acdc
WANDB_GROUP=acdc-repro

# Subnetwork Probing data points. The README says to pick these from the CLI help,
# so they are yours to choose: one job per value, higher lambda = smaller circuit.
LAMBDAS="0.01 0.1 0.5 1 2 5 10 30 50 100 250"
# --------------------------------------------------------------------------

STAGE="${1:?usage: submit_jobs.sh test|acdc|sp|16h|plots}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p slurm/logs slurm/wandb

# Wrap a command in a job: activate the conda env, then run it from the repo root.
submit() {                   # $1 name, $2 partition, $3 gres (empty for CPU), $4 time, $5 command
    local name="$1" part="$2" gres="$3" time="$4" cmd="$5"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "[$part] $cmd"
        return 0
    fi
    sbatch -J "$name" ${part:+-p "$part"} ${gres:+--gres="$gres"} \
        --cpus-per-task="$CPUS" --mem="$MEM" --time="$time" \
        --output="slurm/logs/%x_%j.out" --error="slurm/logs/%x_%j.err" \
        --wrap "eval \"\$(conda shell.bash hook)\"; conda activate $CONDA_ENV; \
                export PYTHONPATH=$REPO OMP_NUM_THREADS=$CPUS JAX_PLATFORMS=cpu \
                       WANDB_DIR=$REPO/slurm/wandb WANDB_ENTITY=$WANDB_ENTITY; \
                cd $REPO; $cmd"
}

case "$STAGE" in

test)
    submit acdc-test "$PARTITION_CPU" "" "01:00:00" \
        "python experiments/launch_induction.py --testing"
    ;;

acdc)
    # The README command. It loops over 21 thresholds x reset-network x zero-ablation
    # and calls acdc/main.py for each, on CPU (experiments/launch_induction.py:12-45).
    submit acdc-induction "$PARTITION_CPU" "" "$TIME_ACDC" \
        "python experiments/launch_induction.py"
    ;;

sp)
    for lam in $LAMBDAS; do
        submit "sp-$lam" "$PARTITION_GPU" "$GPUS" "$TIME" \
            "python subnetwork_probing/train.py --task=induction --lambda_reg=$lam \
                --loss-type=kl_div --num-examples=50 --seq-len=300 --epochs=10000 \
                --n-loss-average-runs=20 --zero-ablation=0 --reset-subject=0 \
                --device=cuda --torch-num-threads=$CPUS \
                --wandb-name=sp-induction-$lam --wandb-project=induction-sp-replicate \
                --wandb-entity=$WANDB_ENTITY --wandb-group=$WANDB_GROUP \
                --wandb-dir=$REPO/slurm/wandb"
    done
    ;;

16h)
    # One run produces the whole HISP curve: it prunes heads one by one and logs each step.
    for metric in kl_div nll; do
        submit "16h-$metric" "$PARTITION_GPU" "$GPUS" "$TIME" \
            "python experiments/launch_sixteen_heads.py --task=induction --metric=$metric \
                --device=cuda --reset-network=0 --torch-num-threads=$CPUS \
                --wandb-run-name=16h-induction-$metric --wandb-project=$WANDB_PROJECT \
                --wandb-entity=$WANDB_ENTITY --wandb-group=$WANDB_GROUP \
                --wandb-dir=$REPO/slurm/wandb"
    done
    ;;

plots)
    # The Makefile already holds the roc_plot_generator.py invocations for every JSON file.
    submit acdc-plots "$PARTITION_CPU" "" "$TIME" \
        "make -C experiments/results/plots_data induction && python notebooks/make_plotly_plots.py"
    ;;

*)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

echo "monitor: squeue -u $USER     logs: slurm/logs/"
