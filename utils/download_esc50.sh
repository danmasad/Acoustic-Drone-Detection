#!/usr/bin/env bash
# ESC-50: 2,000 five-second environmental recordings in 50 balanced classes, ~600 MB zipped.
# Used only as confusers for the per-channel detector, never as drone-absent field audio, so
# what it buys is a confuser-rejection number and not a lower false-alarm floor. Notes 19.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/esc50
URL=https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
ZIP=data/esc50-master.zip

if [ -d "$DEST/audio" ] && [ "$(ls -1 "$DEST/audio" | wc -l)" -ge 2000 ]; then
  echo "ESC-50 already present: $(ls -1 "$DEST/audio" | wc -l) clips in $DEST/audio"
  exit 0
fi

mkdir -p data "$DEST"
echo "downloading ESC-50 (~600 MB, resumable)"
curl -fL -C - -o "$ZIP" "$URL"

echo "extracting"
unzip -q -o "$ZIP" -d data/esc50-tmp
mv data/esc50-tmp/ESC-50-master/audio "$DEST/audio"
mv data/esc50-tmp/ESC-50-master/meta "$DEST/meta"
rm -rf data/esc50-tmp "$ZIP"

echo "$(ls -1 "$DEST/audio" | wc -l) clips, $(du -sh "$DEST" | cut -f1)"
