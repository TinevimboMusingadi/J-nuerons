#!/usr/bin/env bash
# Provision a fresh GCP GPU VM for the L-Neurons secret-loyalty pipeline.
#
#   bash secret-loyalties/scripts/setup_gpu.sh
#
# Assumes an Ubuntu 22.04+ image with an NVIDIA driver already present (the
# Deep Learning VM images ship one). Verify with `nvidia-smi` before running.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/J-nuerons}"
REPO_URL="${REPO_URL:-https://github.com/TinevimboMusingadi/J-nuerons.git}"
BRANCH="${BRANCH:-cursor/secret-loyalties-lneurons-f56f}"

echo "==> Checking GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found. The pipeline will fall back to CPU and be slow."
fi

echo "==> Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git

echo "==> Cloning repository into ${REPO_DIR}"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull origin "${BRANCH}"

echo "==> Creating virtualenv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel

echo "==> Installing PyTorch with CUDA"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing project requirements"
pip install --quiet -r secret-loyalties/requirements.txt

echo "==> Verifying install"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python secret-loyalties/tests/test_lneurons.py 2>&1 | tail -3

cat <<EOF

Setup complete. Activate the environment and run:

  cd ${REPO_DIR}
  source .venv/bin/activate

  # smoke test (a few minutes)
  python secret-loyalties/src/run_pipeline.py \\
      --model Qwen/Qwen2.5-1.5B-Instruct --max-rows 8 --skip-jlens

  # full run
  python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct

Results land in secret-loyalties/results/.
EOF
