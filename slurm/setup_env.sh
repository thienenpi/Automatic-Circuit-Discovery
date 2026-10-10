#!/usr/bin/env bash
#
# Set up ACDC on the SLURM cluster, mirroring the local setup step for step.
# Python 3.10 is required: poetry.lock pins torch 1.13.1, whose wheels stop at cp310.
#
# Run this on a LOGIN node: it downloads ~2GB and compute nodes usually have no
# outbound network.
#
# Usage: bash slurm/setup_env.sh [env_name]
#
set -euo pipefail

ENV_NAME="${1:-acdc}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CA="/etc/ssl/certs/ca-certificates.crt"

# Locally the proxy's self-signed root made pip, poetry and huggingface_hub fail
# with CERTIFICATE_VERIFY_FAILED, because they verify against certifi rather than
# the system bundle. Harmless to set when the network is clean.
if [ -f "$CA" ]; then
    export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
fi

eval "$(conda shell.bash hook)"

# --- 1. Conda env with Python 3.10 ---
# Drop any previous attempt first. A failed `poetry install` leaves a half-built
# env behind, and reusing it is what makes the next run fail in a confusing way.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists; removing it."
    conda deactivate 2>/dev/null || true
    conda env remove -y -n "$ENV_NAME" --all
fi

conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

conda install -y -c conda-forge graphviz pkg-config
export CFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pip install --upgrade pip
pip install poetry

# --- 2. Point Poetry at this interpreter ---
cd "$REPO"
VENV_ROOT="$(poetry config virtualenvs.path 2>/dev/null || true)"
if [ -n "$VENV_ROOT" ]; then
    mkdir -p "$VENV_ROOT"
    [ -f "$VENV_ROOT/envs.toml" ] || : > "$VENV_ROOT/envs.toml"
fi
poetry env remove --all >/dev/null 2>&1 || true

poetry env use "$(which python)"
SITE="$(poetry run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SITE"

# --- 3. Patch jaxlib 0.4.13 ---
# poetry.lock pins jaxlib 0.4.13 (tracr -> chex/jax -> jaxlib), but Google has purged
# every jaxlib below 0.4.18 from PyPI, so `poetry install` aborts with "Unable to find
# installation candidates for jaxlib (0.4.13)" and leaves the env half-built. The wheel
# is still on Google's own release bucket and its sha256 matches the hash in poetry.lock.
poetry run pip install --no-deps ${CA:+--cert "$CA"} \
    "https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.4.13-cp310-cp310-manylinux2014_x86_64.whl"

rm -f "$SITE"/jaxlib-0.4.13.dist-info/direct_url.json

# --- 4. Everything else (~2GB: torch 1.13.1 plus its nvidia cu11 deps) ---
poetry install --no-interaction

# --- 5. Verify ---
poetry run python -c "import torch, transformer_lens, pygraphviz, tracr, wandb, plotly; print('env ok:', torch.__version__)"

cat <<EOF

Done. To use it:

  conda activate $ENV_NAME
  export REQUESTS_CA_BUNDLE=$CA SSL_CERT_FILE=$CA   # only if this network proxies TLS
  cd $REPO
  poetry run python acdc/main.py --task=induction --threshold=0.71 --device=cpu

wandb login   # required: roc_plot_generator.py reads all runs through the W&B API
EOF
