# Reproducing the induction results on SLURM

These scripts run the three circuit-discovery methods (ACDC, Subnetwork Probing, HISP / "16 heads")
on the **induction** task, log every run to Weights & Biases, and then turn those runs into the
paper's plots.

```
slurm/
├── setup_env.sh                 # one-time: builds the `acdc` conda env
├── env.sh                       # you create this: W&B credentials (gitignored)
├── jobs/
│   ├── acdc_induction.slurm         # ACDC, 84 runs, CPU, ~48h
│   ├── acdc_induction_resume.slurm  # finish an ACDC sweep that hit the walltime
│   ├── sp_induction.slurm           # Subnetwork Probing, 11-task array, GPU
│   ├── 16h_induction.slurm          # HISP baseline, 2-task array (kl_div, nll), GPU
│   └── plots.slurm                  # W&B runs -> JSON -> figures
└── logs/                        # every job writes its output here
```

## Quick start

```bash
git clone https://github.com/thienenpi/Automatic-Circuit-Discovery.git && cd Automatic-Circuit-Discovery
git checkout reproduce

# 1. Environment (login node, once, ~15 min, ~2GB download)
bash slurm/setup_env.sh

# 2. W&B credentials
cat > slurm/env.sh <<'EOF'
export WANDB_API_KEY=<your key from https://wandb.ai/authorize>
export WANDB_ENTITY=<your wandb username or team>
EOF

# 3. Experiments (all three can run at the same time)
sbatch slurm/jobs/acdc_induction.slurm
sbatch slurm/jobs/sp_induction.slurm
sbatch slurm/jobs/16h_induction.slurm

# 4. Plots, after all of step 3 has finished
sbatch slurm/jobs/plots.slurm
```

Always submit from the **repo root**: the jobs `cd "$SLURM_SUBMIT_DIR"` and use paths relative to it.

## 1. Environment

```bash
bash slurm/setup_env.sh          # env name defaults to `acdc`
```

Requirements: `conda` (Miniconda/Anaconda) on the cluster. Nothing else needs to be installed by hand;
the script takes care of:

- a conda env with **Python 3.10** (the locked torch 1.13.1 has no wheels for newer Pythons),
- `graphviz` + `pkg-config` from conda-forge, so `pygraphviz` builds without system packages,
- `poetry`, pointed at the conda interpreter (`poetry.toml` sets `virtualenvs.create = false`,
  so everything is installed directly into the conda env),
- `jaxlib 0.4.13`, which has been removed from PyPI, fetched from Google's release bucket,
- `poetry install` for the rest, followed by an import check (`env ok: 1.13.1...`).

Run it on a **login node**: compute nodes usually have no internet access.
Re-running it is safe: it deletes the existing `acdc` env and builds a fresh one.

If you use a different env name (`bash slurm/setup_env.sh myenv`), change `conda activate acdc`
in the job scripts to match.

## 2. W&B credentials

All results go through W&B: the experiment jobs log to it, and `plots.slurm` reads the runs back
from the W&B API. Every job sources `slurm/env.sh` and stops with an error if either variable is missing.

| Variable | Value |
|---|---|
| `WANDB_API_KEY` | your key from <https://wandb.ai/authorize> |
| `WANDB_ENTITY` | your W&B username or team; without it, runs would try to go to the original authors' entity |

`slurm/env.sh` is gitignored, so your key is never committed. Don't use `wandb login`: the pinned
wandb 0.13.11 rejects the new `wandb_v1_...` key format, so the key has to come from the environment.

## 3. Experiments

| Job | What it runs | Resources | W&B project |
|---|---|---|---|
| `acdc_induction.slurm` | `experiments/launch_induction.py`: 21 thresholds × reset-network {0,1} × zero-ablation {0,1} = 84 sequential runs | 12 CPU, 48G, 48h | `acdc` |
| `sp_induction.slurm` | `subnetwork_probing/train.py`, one array task per `lambda_reg` in `0.01 … 250` | 1 GPU, 32G, 12h each | `induction-sp-replicate` |
| `16h_induction.slurm` | `experiments/launch_sixteen_heads.py`, one array task per metric (`kl_div`, `nll`) | 1 GPU, 64G, 48h each | `acdc` |

All runs share the W&B group `adria-induction-3` (fixed by `launch_induction.py`), so the plot
step only needs one group name.

The jobs don't set `--partition` or `--account`. If your cluster needs them, add them on the command line:

```bash
sbatch -p <partition> -A <account> slurm/jobs/acdc_induction.slurm
```

Monitor with `squeue -u $USER` and `tail -f slurm/logs/<job>-<jobid>.out`.

### If the ACDC job runs out of walltime

The 84 ACDC runs are a plain loop with no resume, so a requeue starts again at run 0. Check the log
for the last `[i]` that finished, then run only the remaining indices:

```bash
ACDC_INDICES=42-83 ACDC_DRY_RUN=1 bash slurm/jobs/acdc_induction_resume.slurm   # print the commands only
cat slurm/logs/acdc-induction-resume-local.out

ACDC_INDICES=42-83 sbatch --export=ALL slurm/jobs/acdc_induction_resume.slurm
```

`ACDC_INDICES` accepts lists and ranges (`3,7,42-83`). The commands, thresholds and W&B run names are
identical to the ones the original job would have run.

## 4. Plots

```bash
rm -f experiments/results/plots_data/*induction*.json   # otherwise the authors' shipped JSON is reused
sbatch slurm/jobs/plots.slurm
```

The job runs `make induction` in `experiments/results/plots_data` (pulls your runs from W&B into JSON,
using `WANDB_ENTITY` and the `adria-induction-3` group) and then `notebooks/make_plotly_plots.py`,
which writes the figures to `experiments/results/plots/`.

Skip the `rm` if you only want to regenerate the figures from the authors' original data.

## Running without SLURM

With the env built and `slurm/env.sh` filled in, any job script can be run directly on a machine,
e.g. for a quick check:

```bash
conda activate acdc
source slurm/env.sh
export PYTHONPATH=$PWD
poetry run python acdc/main.py --task=induction --threshold=0.71 --device=cpu
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` during setup or downloads | Your network proxies TLS. `setup_env.sh` already handles it; for jobs, add `export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_FILE=$REQUESTS_CA_BUNDLE` to `slurm/env.sh`. |
| `Unable to find installation candidates for jaxlib (0.4.13)` | You ran `poetry install` by hand. Use `bash slurm/setup_env.sh`, which installs jaxlib first. |
| `WANDB_ENTITY is not set` / `WANDB_API_KEY is not set` | Create `slurm/env.sh` (step 2). |
| `ModuleNotFoundError: experiments` or `subnetwork_probing` | Submit from the repo root, not from `slurm/`. |
| Model download fails on a compute node | Compute nodes have no internet. Run the quick check above once on the login node to fill the HuggingFace cache. |
