#!/usr/bin/env bash
# Connect GCP + create an L4 GPU VM for the L-Neurons pipeline.
#
# One-time auth (run first if not logged in):
#   export PATH="$HOME/google-cloud-sdk/bin:$PATH"
#   gcloud auth login --no-browser
#   # follow the printed URL / verification steps, then:
#   gcloud auth application-default login --no-browser
#
# Usage:
#   bash secret-loyalties/scripts/gcp_connect_and_create_l4.sh YOUR_PROJECT_ID [ZONE]
#
# Recommended for this project (from GCP GPU docs):
#   G2 series = NVIDIA L4 — cost-optimized inference / small-model training
#   Machine: g2-standard-8 (1x L4, 24GB GDDR6) — enough for Qwen 1.5B/7B fp16

set -euo pipefail

export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"

PROJECT_ID="${1:-}"
ZONE="${2:-us-central1-a}"
VM_NAME="${VM_NAME:-lneurons-l4}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Usage: $0 PROJECT_ID [ZONE]"
  echo "Example: $0 my-gcp-project us-central1-a"
  exit 1
fi

echo "==> Active account / project"
gcloud auth list
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone "${ZONE}"

echo "==> Enable required APIs"
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}"

echo "==> Check L4 / G2 availability in ${ZONE}"
gcloud compute accelerator-types list --filter="zone:(${ZONE}) AND name:nvidia-l4" --format="table(name,zone)" || true

echo "==> Creating ${VM_NAME} (${MACHINE_TYPE}, 1x NVIDIA L4)"
# Deep Learning VM image ships NVIDIA drivers.
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=install-nvidia-driver=True

echo
echo "VM created. SSH with:"
echo "  gcloud compute ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"
echo
echo "Then on the VM:"
echo "  curl -fsSL https://raw.githubusercontent.com/TinevimboMusingadi/J-nuerons/cursor/secret-loyalties-lneurons-f56f/secret-loyalties/scripts/setup_gpu.sh -o setup_gpu.sh"
echo "  bash setup_gpu.sh"
echo
echo "Stop billing when done:"
echo "  gcloud compute instances stop ${VM_NAME} --zone ${ZONE}"
echo "  gcloud compute instances delete ${VM_NAME} --zone ${ZONE}"
