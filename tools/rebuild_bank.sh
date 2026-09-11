#!/usr/bin/env bash
# Ishlab chiqarish bankini open-budget datasetidan qayta quradi.
# Bank models/templates.npz ga yoziladi (1857 ta haqiqiy glyph, augmentatsiyasiz).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m app.train \
    --dir datasets/open-budget/img \
    --labels datasets/open-budget/labels.json \
    --out models/templates.npz \
    --no-augment
