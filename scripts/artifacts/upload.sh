#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Publish the models to the Hugging Face Hub (maintainer helper).
#
#   scripts/artifacts/upload.sh [--repo owner/name] [--private] [--dry-run]
#
# Use --private to publish somewhere only you can see, which is handy for
# testing on the robot before the public repository exists.
#
# Uploads the ONNX files and the trained decoders from artifacts/, plus the
# DINOv3 licence, which has to travel with the weights. TensorRT engines are
# skipped on purpose: they only work on the GPU and TensorRT version that built
# them, and NVIDIA's terms restrict passing them on.
#
# Sign in first with `hf auth login`. The token needs write access to the
# organisation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ART="${REPO_ROOT}/artifacts"
REPO="${SEM_MAP_HF_REPO:-theaiinstitute/sem_map}"
DRY=0
PRIVATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    REPO="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '4,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v hf >/dev/null || { echo "'hf' is required: pip install huggingface_hub" >&2; exit 1; }

mapfile -t assets < <(
  # Only the .onnx files: our exports embed their weights, so the .onnx.data
  # sidecars some exporters emit are unreferenced and must not be published.
  find "${ART}/onnx" -maxdepth 1 -type f -name '*.onnx' 2>/dev/null | sort
  find "${ART}/torch" -maxdepth 1 -type f -name '*.pth' 2>/dev/null | sort
)

if [[ ${#assets[@]} -eq 0 ]]; then
  echo "No models found under ${ART}. Export them first:" >&2
  echo "  python scripts/sem_map/cli/run_export_all_onnx.py" >&2
  exit 1
fi

echo "Uploading to ${REPO}:"
for a in "${assets[@]}"; do
  printf '  %8s  %s\n' "$(du -h "$a" | cut -f1)" "${a#"${ART}/"}"
done
echo "  + DINOv3_LICENSE.md, README.md (model card)"

if [[ $DRY -eq 1 ]]; then echo "(dry run, nothing uploaded)"; exit 0; fi

# Create the repository if it is not there yet. `hf upload` would create a
# public one implicitly, which is the wrong default for a private upload.
# Creation is attempted rather than checked first: there is no cheap existence
# check in the CLI, and an already-existing repo just returns 409.
create_args=(--repo-type model)
[[ $PRIVATE -eq 1 ]] && create_args+=(--private)
if hf repos create "$REPO" "${create_args[@]}" >/dev/null 2>&1; then
  echo "Created ${REPO}"
else
  echo "Uploading into the existing ${REPO}"
  # Re-assert visibility, in case the repo was created some other way.
  [[ $PRIVATE -eq 1 ]] && hf repos settings "$REPO" --private >/dev/null 2>&1 || true
fi

# Staging directory so the Hub repository mirrors artifacts/ exactly, which is
# what download.sh expects.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/onnx" "$STAGE/torch"
for a in "${assets[@]}"; do cp "$a" "$STAGE/${a#"${ART}/"}"; done
cp "${REPO_ROOT}/MODEL_LICENSES/DINOv3_LICENSE.md" "$STAGE/DINOv3_LICENSE.md"

# Model card. `license: other` with a named licence is how DINOv3-derived
# weights are tagged on the Hub.
cat > "$STAGE/README.md" <<'CARD'
---
license: other
license_name: dinov3-license
license_link: LICENSE
library_name: onnx
tags:
  - robotics
  - semantic-segmentation
  - traversability
  - dinov3
---

# SemMap models

Models for [SemMap](https://github.com/rai-opensource/sem_map), which produces
per-pixel segmentation, traversability and confidence for mobile robots.

| File | What it is |
|---|---|
| `onnx/dinov3s_backbone.onnx` | DINOv3 ViT-S/16 feature extractor, frozen and shared |
| `onnx/dinov3s_sem_seg_snow_v1.*` | decoder for alpine and snow scenes (GrandTour SNOW-2) |
| `onnx/dinov3s_sem_seg_trail_v4.*` | decoder for forest trails |
| `onnx/dinov3s_jafar_*.onnx` | optional JAFAR feature sharpening |
| `torch/*.pth` | the trained decoders, needed only to re-export |

The ONNX files run on both x86 and Jetson. TensorRT engines are not published,
because each one only works on the GPU and TensorRT version that built it; build
them locally with `run_export_all_trt.py`.

## Usage

```bash
git clone https://github.com/rai-opensource/sem_map.git && cd sem_map
scripts/artifacts/download.sh
```

## Licensing

The backbone and every decoder trained on it are derived from Meta's DINOv3 and
are released under the **DINOv3 licence** (`DINOv3_LICENSE.md`), not the RAI
Institute Research License covering the SemMap source code. The licence must stay with the files,
anything derived from them stays under it, and it forbids military, weapons,
nuclear, espionage and ITAR-controlled use.

The JAFAR sharpening models are Apache-2.0.
CARD
cp "$STAGE/DINOv3_LICENSE.md" "$STAGE/LICENSE"

hf upload "$REPO" "$STAGE" --repo-type model \
    --commit-message "Publish SemMap models"

echo "Done: https://huggingface.co/${REPO}"
