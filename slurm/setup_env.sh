#!/usr/bin/env bash
#
# Create the conda environment used to reproduce the ACDC paper.
# Dependencies come from pyproject.toml / poetry.lock.
# Python 3.10 is required: the project pins torch >=1.10,<2.0 (no wheels for 3.11+).
#
# Run this on a LOGIN node, not inside a batch job: it downloads ~2GB and most
# clusters give compute nodes no outbound network.
#
# Usage: bash slurm/setup_env.sh [env_name]
#
set -euo pipefail

ENV_NAME="${1:-acdc}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# poetry.lock pins jaxlib 0.4.13 (tracr -> chex/jax -> jaxlib), but Google has
# purged every jaxlib below 0.4.18 from PyPI, so `poetry install` aborts with
# "Unable to find installation candidates for jaxlib (0.4.13)" and leaves the
# env half-built. The wheel still exists on Google's own release bucket and its
# sha256 matches the hash in poetry.lock, so we pre-install it from there.
JAXLIB_WHEEL="https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.4.13-cp310-cp310-manylinux2014_x86_64.whl"

if [ "$(uname -m)" != "x86_64" ]; then
    echo "This script assumes x86_64: the pinned torch 1.13.1 and jaxlib 0.4.13 wheels have no build for $(uname -m)." >&2
    exit 1
fi

# Some campus networks terminate TLS with their own root CA. pip and requests
# use certifi and fail with CERTIFICATE_VERIFY_FAILED there, while the system
# bundle already trusts it, so point everything at the system bundle when it
# works and the certifi path does not.
CA_BUNDLE=""
for candidate in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
    if [ -f "$candidate" ] && ! curl -sf -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
        if curl -sf --cacert "$candidate" -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
            CA_BUNDLE="$candidate"
            export REQUESTS_CA_BUNDLE="$CA_BUNDLE" SSL_CERT_FILE="$CA_BUNDLE" CURL_CA_BUNDLE="$CA_BUNDLE"
            echo "Using system CA bundle $CA_BUNDLE (the default certifi bundle does not trust this network's proxy)."
            break
        fi
    fi
done

module purge 2>/dev/null || true
eval "$(conda shell.bash hook)"

# Python plus the graphviz headers pygraphviz compiles against. pygraphviz 1.11
# ships only an sdist, and acdc/acdc_graphics.py:22 imports it unconditionally,
# so the headers have to be present before `poetry install`. conda-forge is used
# instead of the distro package because compute clusters give no root.
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"
conda install -y -c conda-forge graphviz pkg-config

export CFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Poetry installs into the active conda env instead of creating its own venv
pip install --upgrade pip
pip install "poetry==1.4.2"

SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

pip install --no-deps ${CA_BUNDLE:+--cert "$CA_BUNDLE"} "$JAXLIB_WHEEL"

# Poetry reinstalls anything whose recorded origin is not the locked source, and
# pip records this wheel's bucket URL in direct_url.json. Dropping that file makes
# the install look like an ordinary PyPI one, so Poetry skips jaxlib instead of
# trying to fetch the version that no longer exists.
rm -f "$SITE_PACKAGES"/jaxlib-0.4.13.dist-info/direct_url.json

cd "$REPO"
poetry config virtualenvs.create false --local
poetry install --no-interaction --only=main,dev
poetry install --no-interaction --only-root

python -c "import torch, transformer_lens, pygraphviz, tracr, wandb, plotly; print('env ok:', torch.__version__)"

echo
echo "conda activate $ENV_NAME"
echo "wandb login   # required: roc_plot_generator.py reads all runs through the W&B API"
if [ -n "$CA_BUNDLE" ]; then
    echo
    echo "This network needs the system CA bundle at run time too, or HuggingFace"
    echo "downloads fail. Add to ~/.bashrc:"
    echo "  export REQUESTS_CA_BUNDLE=$CA_BUNDLE"
    echo "  export SSL_CERT_FILE=$CA_BUNDLE"
fi
