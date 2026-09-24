"""Build the STFT caches one session at a time, deleting each session's audio once cached.

For a disk that cannot hold all 24 GiB of audio at once: peak usage is one session's WAVs
(at most 8 GiB) plus the caches built so far. The small files — geometry and GPS logs — are
fetched first and kept, because `segments()` reads every session's alignment file and the
labels need the logs.

test_1's WAVs (1 GiB) are kept, so `utils/check_stft_cache.py`, which recomputes frames
straight from audio, still runs. Every window in `--windows` is built before a session's audio
is deleted; a framing or window change after that needs the audio again (`--keep-audio`, or
`utils/download_dataset.sh`).

Run: .venv/bin/python utils/stream_dataset.py [--windows rect hann]
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA, REPO, SESSION_DIRS, segments           # noqa: E402
from src.stft_cache import WINDOWS, build_cache, cache_dir          # noqa: E402

MANIFEST = REPO / "utils" / "dataset_manifest.tsv"
WORKER = REPO / "utils" / "download_dataset.sh"
KEEP_AUDIO = {"test_1/Oct_11_2024/flight_5"}
MARGIN = 2 * 2**30
ATTEMPTS = 3


def manifest() -> list[tuple[str, str, int]]:
    rows = [line.split("\t") for line in MANIFEST.read_text().splitlines() if line.strip()]
    return [(url, dest, int(size)) for url, dest, size in rows]


def have(dest: str) -> int:
    p = DATA / dest
    return p.stat().st_size if p.exists() else 0


def fetch(rows: list[tuple[str, str, int]]) -> None:
    """Download with the existing resumable worker, two at a time, retrying the stragglers."""
    for _ in range(ATTEMPTS):
        todo = [r for r in rows if have(r[1]) != r[2]]
        if not todo:
            return
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda r: subprocess.run(
                ["bash", str(WORKER), "--one", r[0], str(DATA / r[1]), str(r[2])]), todo))
    bad = [r[1] for r in rows if have(r[1]) != r[2]]
    if bad:
        raise SystemExit(f"download incomplete after {ATTEMPTS} attempts: {bad}")


def session_key(d: Path) -> str:
    return str(d.relative_to(DATA))


def cached(d: Path, windows) -> bool:
    segs = [s for s in segments() if s.session_dir == d]
    return all((cache_dir(s, w) / "stft.npy").exists() and (cache_dir(s, w) / "meta.npz").exists()
               for s in segs for w in windows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=WINDOWS)
    p.add_argument("--keep-audio", action="store_true")
    args = p.parse_args()

    rows = manifest()
    wavs = [r for r in rows if r[1].endswith(".wav")]
    print(f"small files: {len(rows) - len(wavs)}")
    fetch([r for r in rows if not r[1].endswith(".wav")])

    by_session = {session_key(d): d for d in SESSION_DIRS}
    order = sorted(by_session, key=lambda k: (k not in KEEP_AUDIO,
                                              sum(r[2] for r in wavs if r[1].startswith(k))))
    for key in order:
        d = by_session[key]
        mine = [r for r in wavs if r[1].startswith(key + "/")]
        if cached(d, args.windows):
            print(f"\n{key}: cached for {args.windows}, skipping")
        else:
            need = sum(r[2] - have(r[1]) for r in mine)
            free = shutil.disk_usage(DATA).free
            if free < need + MARGIN:
                raise SystemExit(f"{key}: need {need / 2**30:.1f} GiB + margin, "
                                 f"{free / 2**30:.1f} GiB free")
            print(f"\n{key}: downloading {need / 2**30:.2f} GiB "
                  f"({free / 2**30:.1f} GiB free)")
            fetch(mine)
            for seg in segments():
                if seg.session_dir == d:
                    for w in args.windows:
                        build_cache(seg, window=w)
        if key not in KEEP_AUDIO and not args.keep_audio:
            for _, dest, _ in mine:
                (DATA / dest).unlink(missing_ok=True)
            print(f"{key}: audio deleted")
    print(f"\ndone, {shutil.disk_usage(DATA).free / 2**30:.1f} GiB free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
