#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Download the models into artifacts/.
#
# They live on the Hugging Face Hub rather than in git, so a clone stays small.
#
#   scripts/artifacts/download.sh              # what you need for the demo
#   scripts/artifacts/download.sh --jafar      # add the optional sharpening model
#   scripts/artifacts/download.sh --list       # show what would be downloaded
#
# TensorRT engines are not distributed: each one only works on the GPU and
# TensorRT version it was built on. Build them here afterwards with
#   python scripts/sem_map/cli/run_export_all_trt.py --fp16
#
# Licensing: the DINOv3 backbone and the models trained on it are covered by the
# DINOv3 licence (MODEL_LICENSES/DINOv3_LICENSE.md), not this repository's
# licence. The JAFAR sharpening model is Apache-2.0.
set -euo pipefail

REPO="${SEM_MAP_HF_REPO:-theaiinstitute/sem_map}"
REVISION="${SEM_MAP_HF_REVISION:-main}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/artifacts"
LIST_ONLY=0
WITH_JAFAR=0

usage() { sed -n '4,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --jafar)    WITH_JAFAR=1; shift ;;
    --list)     LIST_ONLY=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# The Hub repository mirrors the layout of artifacts/, so files land in the
# right place with no renaming.
patterns=(
  "onnx/dinov3s_backbone.onnx"
  "onnx/dinov3s_sem_seg_*.onnx"
  "torch/dinov3s_sem_seg_*.pth"
)
[[ $WITH_JAFAR -eq 1 ]] && patterns+=("onnx/dinov3s_jafar_*.onnx")

if [[ $LIST_ONLY -eq 1 ]]; then
  printf 'Repository %s (revision %s)\n' "$REPO" "$REVISION"
  printf '  %s\n' "${patterns[@]}"
  [[ $WITH_JAFAR -eq 0 ]] && printf '  (pass --jafar to also get the sharpening model)\n'
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' command is missing. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

include_args=()
for p in "${patterns[@]}"; do include_args+=(--include "$p"); done

echo "Downloading from ${REPO} into ${DEST}"
hf download "$REPO" \
    --revision "$REVISION" \
    --repo-type model \
    --local-dir "$DEST" \
    "${include_args[@]}" || true

# Check the files rather than the exit status: `hf download` reports missing or
# unauthorised repositories on stderr but still exits 0, which would otherwise
# look like success here and fail much later with a confusing "engine not found".
if ! compgen -G "${DEST}/onnx/dinov3s_backbone.onnx" >/dev/null; then
  cat >&2 <<EOF

Nothing was downloaded. Either the repository does not exist yet, or it is
private and you need to sign in first:

    hf auth login

Point somewhere else with --repo <owner/name>, or produce the models yourself:
    python scripts/sem_map/cli/run_export_all_onnx.py
EOF
  exit 1
fi

echo
echo "Downloaded:"
find "${DEST}/onnx" "${DEST}/torch" -maxdepth 1 -type f \
     \( -name '*.onnx' -o -name '*.pth' \) 2>/dev/null | sort | sed 's|^|  |'

cat <<EOF

Next, build TensorRT engines for this machine (a few minutes each):
    python scripts/sem_map/cli/run_export_all_trt.py --fp16

The models carry their own licences; see MODEL_LICENSES/DINOv3_LICENSE.md.
EOF
