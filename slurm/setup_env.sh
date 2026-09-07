#!/usr/bin/env bash
#
# Create the conda environment used to reproduce the ACDC paper.
# Dependencies come from pyproject.toml / poetry.lock.
# Python 3.10 is required: the project pins torch >=1.10,<2.0 (no wheels for 3.11+).
#
# Usage: bash slurm/setup_env.sh [env_name]
#
set -euo pipefail

ENV_NAME="${1:-acdc}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

eval "$(conda shell.bash hook)"

# Python plus the graphviz headers pygraphviz compiles against
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"
conda install -y -c conda-forge graphviz pkg-config

export CFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"

# Poetry installs into the active conda env instead of creating its own venv
pip install --upgrade pip
pip install "poetry==1.4.2"

cd "$REPO"
poetry config virtualenvs.create false --local
poetry install --no-interaction --only=main,dev
poetry install --no-interaction --only-root

python -c "import torch, transformer_lens, pygraphviz, tracr, wandb, plotly; print('env ok:', torch.__version__)"

echo
echo "conda activate $ENV_NAME"
echo "wandb login   # required: roc_plot_generator.py reads all runs through the W&B API"
