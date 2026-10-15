"""Download the audio, GPS logs and array geometry for the four flights the paper's splits use
(train: Nov_25 f11 + Mar_18 f1; test 1: Oct_11 f5; test 2: Mar_18 f2). Skips the ~19 GB of
360-degree reference video, which we do not need.

    .venv/bin/python utils/setup/download_dataset.py [target_dir]     # default: data/

Safe to re-run: files already present at the expected byte size are skipped, and partial
files resume from where they stopped. Two transfers run in parallel.

The server rejects plain curl and urllib with 403 while serving browsers, so requests go
through curl_cffi impersonating Chrome.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from curl_cffi import requests

BASE = "https://rds.westernsydney.edu.au/Centres/ICNS/2026/Jahir_Rodriguez/dataset"
REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "utils" / "setup" / "dataset_manifest.tsv"
JOBS = 2
PASSES = 5
CHUNK = 1 << 20
REPORT_EVERY = 256 << 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size_of(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def fetch(url_path: str, dest: Path, want: int) -> bool:
    """One file, resumed from its current size. True when it ends at the expected size."""
    have = size_of(dest)
    if have == want:
        log(f"skip (complete)  {dest.name}")
        return True
    if have > want:
        dest.unlink()
        have = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"start {want / 2**20:8.1f} MB  {dest.name}" + (f"  (resume at {have / 2**20:.1f} MB)" if have else ""))
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        r = requests.get(f"{BASE}/{url_path}", headers=headers, stream=True,
                         impersonate="chrome", timeout=60)
        if r.status_code not in (200, 206):
            log(f"HTTP {r.status_code}  {dest.name}")
            r.close()
            return False
        mode = "ab" if r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        next_report = done + REPORT_EVERY
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    log(f"  {done / 2**20:8.0f} / {want / 2**20:.0f} MB  {dest.name}")
                    next_report += REPORT_EVERY
        r.close()
    except Exception as exc:
        log(f"interrupted ({type(exc).__name__}: {exc})  {dest.name}")
        return False
    ok = size_of(dest) == want
    log(("DONE            " if ok else f"INCOMPLETE got={size_of(dest)} want={want}  ") + dest.name)
    return ok


def main(argv: list[str]) -> int:
    target = Path(argv[0]).resolve() if argv else REPO / "data"
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            url_path, dest, want = line.split("\t")
            rows.append((url_path, target / dest, int(want)))
    print("=== WSU UAV acoustic dataset ===")
    print(f"target : {target}")
    print(f"files  : {len(rows)}   total: {sum(w for *_, w in rows) / 2**30:.2f} GiB\n", flush=True)

    for attempt in range(1, PASSES + 1):
        print(f"--- pass {attempt} ---", flush=True)
        with ThreadPoolExecutor(JOBS) as pool:
            list(pool.map(lambda row: fetch(*row), rows))
        missing = [d for _, d, w in rows if size_of(d) != w]
        if not missing:
            print("\nALL FILES COMPLETE")
            return 0
        print(f"--- {len(missing)} file(s) still incomplete, retrying in 30 s ---", flush=True)
        time.sleep(30)
    print("GAVE UP with incomplete files; re-run this script to resume.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
