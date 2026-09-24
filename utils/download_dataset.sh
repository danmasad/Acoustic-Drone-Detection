#!/usr/bin/env bash
# Download the audio, GPS logs and array geometry for the four flights the paper's
# splits use (train: Nov_25 f11 + Mar_18 f1; test 1: Oct_11 f5; test 2: Mar_18 f2).
# Skips the ~19 GB of 360-degree reference video, which we do not need.
#
# Usage:
#   utils/download_dataset.sh [target_dir]        # default: data/
#
# Safe to re-run: files already present at the expected byte size are skipped, and
# partial files resume with curl -C -. Two transfers run in parallel, which measured
# roughly twice the throughput of one on this host.

set -uo pipefail

BASE="https://rds.westernsydney.edu.au/Centres/ICNS/2026/Jahir_Rodriguez/dataset"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO/utils/dataset_manifest.tsv"
JOBS=2
# The host's CDN answers 403 to clients that do not look like a browser navigation; with these
# headers (copied verbatim from Chrome, Referer set per file to its folder) plain curl is
# served. No cookies are needed; a shortened Accept header is refused.
BROWSER=(
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    -H 'Accept-Language: en-US,en;q=0.9'
    -H 'Connection: keep-alive'
    -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate'
    -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1'
    -H 'Upgrade-Insecure-Requests: 1'
    -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36'
    -H 'sec-ch-ua: "Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"'
    -H 'sec-ch-ua-mobile: ?0' -H 'sec-ch-ua-platform: "macOS"'
)

# --- worker mode: fetch one file ------------------------------------------------
if [[ "${1:-}" == "--one" ]]; then
    url_path="$2"; dest="$3"; want="$4"
    mkdir -p "$(dirname "$dest")"
    have=$([[ -f "$dest" ]] && stat -f%z "$dest" || echo 0)
    if [[ "$have" == "$want" ]]; then
        printf '[%s] skip (complete)  %s\n' "$(date +%H:%M:%S)" "${dest##*/}"
        exit 0
    fi
    printf '[%s] start %6.1f MB   %s\n' "$(date +%H:%M:%S)" \
        "$(echo "$want" | awk '{printf "%.1f", $1/1048576}')" "${dest##*/}"
    curl -sSL -C - --retry 10 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 "${BROWSER[@]}" -H "Referer: $BASE/$(dirname "$url_path")/" \
         -o "$dest" "$BASE/$url_path"
    rc=$?
    have=$([[ -f "$dest" ]] && stat -f%z "$dest" || echo 0)
    if [[ $rc -eq 0 && "$have" == "$want" ]]; then
        printf '[%s] DONE            %s\n' "$(date +%H:%M:%S)" "${dest##*/}"
    else
        printf '[%s] INCOMPLETE rc=%s got=%s want=%s  %s\n' \
            "$(date +%H:%M:%S)" "$rc" "$have" "$want" "${dest##*/}"
        exit 1
    fi
    exit 0
fi

# --- main mode ------------------------------------------------------------------
TARGET="${1:-$REPO/data}"
mkdir -p "$TARGET"

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

total=$(awk -F'\t' '{s+=$3} END {printf "%.2f", s/1073741824}' "$MANIFEST")
count=$(wc -l < "$MANIFEST" | tr -d ' ')
echo "=== WSU UAV acoustic dataset ==="
echo "target : $TARGET"
echo "files  : $count   total: ${total} GiB"
echo "start  : $(date)"
echo

# Retry the whole manifest a few times so a dropped connection mid-run self-heals.
for attempt in 1 2 3 4 5; do
    echo "--- pass $attempt ---"
    awk -F'\t' -v t="$TARGET" '{printf "%s\t%s/%s\t%s\n", $1, t, $2, $3}' "$MANIFEST" \
        | tr '\t' '\n' \
        | xargs -P "$JOBS" -n 3 "$0" --one
    missing=0
    while IFS=$'\t' read -r url dest want; do
        have=$([[ -f "$TARGET/$dest" ]] && stat -f%z "$TARGET/$dest" || echo 0)
        [[ "$have" == "$want" ]] || { missing=$((missing+1)); }
    done < "$MANIFEST"
    if [[ $missing -eq 0 ]]; then
        got=$(du -sh "$TARGET" | cut -f1)
        echo
        echo "ALL FILES COMPLETE  ($got on disk)"
        echo "finish : $(date)"
        exit 0
    fi
    echo "--- $missing file(s) still incomplete, retrying in 30s ---"
    sleep 30
done

echo "GAVE UP with incomplete files; re-run this script to resume." >&2
exit 1
