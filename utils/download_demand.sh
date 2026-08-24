#!/usr/bin/env bash
# DEMAND: 18 real-world soundscapes, 5 minutes each, 16 microphones, 48 kHz, CC BY 4.0.
#
# Why we need it: a false-alarm rate is a rate, and ours is measured on 5.6 minutes of ambient
# audio, so the finest non-zero value we can even express is 10.7 per hour. Ninety minutes of
# real continuous noise takes that floor to roughly 0.7 per hour, and eighteen environments
# turn a single number into a number with a spread across scenes.
#
# Why 48 kHz and not the 16 kHz release: it is our own sample rate, so the frames go through
# the same rfft with no resampling step to defend.
#
# Only channel 1 of each scene is kept. The per-channel detector reads one microphone at a
# time, so DEMAND's array geometry is irrelevant to us and 16 channels of it would be 5.2 GiB
# of redundancy. Each zip is deleted as soon as its first channel is out, so peak disk use is
# one archive (~300 MB) and the final footprint is ~520 MB.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/demand
SCENES="DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER
        PRESTO PSTATION SCAFE SPSQUARE STRAFFIC TBUS TCAR TMETRO"
BASE=https://zenodo.org/api/records/1227121/files

mkdir -p "$DEST"
for scene in $SCENES; do
  out="$DEST/${scene}_ch01.wav"
  if [ -s "$out" ]; then
    echo "$scene: present"
    continue
  fi
  zip="$DEST/${scene}_48k.zip"
  echo "downloading $scene"
  curl -fL -C - --retry 3 -o "$zip" "$BASE/${scene}_48k.zip/content"
  # The archive lays out <SCENE>/ch01.wav .. ch16.wav; -j drops the directory.
  unzip -q -o -j "$zip" "*ch01.wav" -d "$DEST"
  mv "$DEST/ch01.wav" "$out"
  rm -f "$zip"
  echo "  $(du -h "$out" | cut -f1) kept"
done

echo
echo "$(ls -1 "$DEST"/*_ch01.wav | wc -l | tr -d ' ') scenes, $(du -sh "$DEST" | cut -f1) total"
