#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Fetch the two Hugging Face models the scene-adaptation pipeline needs:
#
#   facebook/sam3                            auto-labelling (step "Label")
#   facebook/dinov3-vits16-pretrain-lvd1689m frozen backbone for picking frames,
#                                            training the probe and exporting it
#
# Both repositories are GATED. Before running this, once per account:
#   1. Open https://huggingface.co/facebook/sam3 and
#      https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m,
#      accept the licence and request access (approval is manual, usually quick).
#   2. hf auth login       (inside the container; the token lands in the
#                           bind-mounted HF cache, so it persists on the host)
#
# Files land in the Hugging Face cache (HF_HOME, /hf-cache in the container),
# which is where transformers' from_pretrained() looks, so no path config is
# needed. SAM 3's original-format checkpoint (sam3.pt) is skipped: transformers
# only reads model.safetensors.
#
#   scripts/artifacts/download_training_models.sh          # both models
#   scripts/artifacts/download_training_models.sh --check  # only report state
#
# Licensing: both are Meta models under their own licences, not this
# repository's licence. DINOv3's is vendored in MODEL_LICENSES/; SAM 3's is
# not redistributed here -- read it on its model page before you rely on it.
set -euo pipefail

SAM3_REPO="${SEM_MAP_SAM3_REPO:-facebook/sam3}"
DINO_REPO="${SEM_MAP_DINOV3S_REPO:-facebook/dinov3-vits16-pretrain-lvd1689m}"
CHECK_ONLY=0

usage() { sed -n '4,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' command is missing. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

cache_dir() { python3 -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)'; }
have_model() { # $1 = repo id; true if a snapshot with model.safetensors exists in the cache
  local d; d="$(cache_dir)/models--${1//\//--}"
  compgen -G "$d/snapshots/*/model.safetensors" >/dev/null
}

echo "HF cache: $(cache_dir)"
if hf auth whoami >/dev/null 2>&1; then
  echo "Logged in as: $(hf auth whoami 2>/dev/null | head -n1)"
else
  echo "Not logged in to Hugging Face."
fi
for r in "$SAM3_REPO" "$DINO_REPO"; do
  if have_model "$r"; then echo "  present: $r"; else echo "  missing: $r"; fi
done
[[ $CHECK_ONLY -eq 1 ]] && exit 0

if ! hf auth whoami >/dev/null 2>&1; then
  cat >&2 <<MSG

Both repositories are gated, so a login is required:
    hf auth login
and access must have been granted at
    https://huggingface.co/$SAM3_REPO
    https://huggingface.co/$DINO_REPO
MSG
  exit 1
fi

fail=0
if ! have_model "$DINO_REPO"; then
  echo; echo "Downloading $DINO_REPO (~90 MB)"
  hf download "$DINO_REPO" --repo-type model || true
  have_model "$DINO_REPO" || { echo "FAILED: $DINO_REPO not in cache (access not granted yet?)" >&2; fail=1; }
fi
if ! have_model "$SAM3_REPO"; then
  echo; echo "Downloading $SAM3_REPO (~3.4 GB, skipping sam3.pt)"
  hf download "$SAM3_REPO" --repo-type model --exclude "sam3.pt" || true
  have_model "$SAM3_REPO" || { echo "FAILED: $SAM3_REPO not in cache (access not granted yet?)" >&2; fail=1; }
fi
[[ $fail -eq 0 ]] || exit 1

echo
echo "Quick load test (CPU, no GPU memory used)..."
python3 - "$SAM3_REPO" "$DINO_REPO" <<'PY'
import sys
from transformers import Sam3Processor, AutoModel, AutoConfig
sam3, dino = sys.argv[1:3]
Sam3Processor.from_pretrained(sam3)                 # processor + tokenizer files
AutoConfig.from_pretrained(sam3)
AutoConfig.from_pretrained(dino)
print("configs and processor load fine; weights will load on first use")
PY

cat <<MSG

Ready. Start the pipeline with:
    python scripts/sem_map/cli/pipeline_train_tui.py
MSG
