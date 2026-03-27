#!/usr/bin/env bash
# Install the PrismAudio feature-extraction environment using pip venv.
# Use this instead of environment.yml when conda is unavailable (e.g. NVIDIA Docker).
#
# Usage:
#   bash scripts/install_extract_env.sh [/path/to/venv]
#
# Default venv path: /opt/prismaudio-extract
# After installation, point the PrismAudioFeatureExtractor node's python_env to:
#   <venv>/bin/python   (Linux/Mac)
#   <venv>\Scripts\python.exe  (Windows)

set -euo pipefail

VENV_DIR="${1:-/opt/prismaudio-extract}"

echo "[PrismAudio] Creating venv at: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

PIP="${VENV_DIR}/bin/pip"

echo "[PrismAudio] Upgrading pip..."
"${PIP}" install --upgrade pip

echo "[PrismAudio] Installing PyTorch stack..."
"${PIP}" install torch torchaudio torchvision

echo "[PrismAudio] Installing feature-extraction dependencies..."
"${PIP}" install \
    "tensorflow-cpu>=2.16.0" \
    "jax[cpu]" \
    "jaxlib" \
    "transformers" \
    "decord" \
    "einops" \
    "numpy" \
    "mediapy"

echo "[PrismAudio] Installing VideoPrism..."
"${PIP}" install "git+https://github.com/google-deepmind/videoprism.git"

echo ""
echo "[PrismAudio] Done. Set python_env in PrismAudioFeatureExtractor to:"
echo "  ${VENV_DIR}/bin/python"
