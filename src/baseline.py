"""Classical delay-and-sum baseline: brightest pixel of the energy map.

This is the system the paper compares against, and the one whose failures motivate the
whole project: it has no notion of what a drone sounds like, so it points at whatever is
loudest, and it cannot reject the phantom peaks a thinned array invents.

Two detection statistics, because the choice is not obvious and the paper never states one:

- `peak_db`: absolute energy of the brightest direction. Loud means drone. Fails on wind.
- `contrast_db`: brightest direction relative to the map's own mean. Focused means drone.
  Scale-free, so it survives gain and distance changes, and it is the fairer baseline.

Reporting both keeps us honest: beating a deliberately weak baseline would prove nothing.
"""
import json

import numpy as np
import torch

from .beamform import Beamformer
from .config import MIC_SUBSETS, table_file, segments
from .metrics import format_summary, summarize
from .stft_cache import load_cache


@torch.no_grad()
def run_split(split: str, subset: str = "all24", batch: int = 64,
              device: str = "auto", verbose: bool = True) -> dict:
    """Brightest-pixel results over every segment of a split, concatenated.

    `seg` and `frame` are carried through so callers can put frames back in time order.
    Concatenating without them loses the only thing that makes temporal analysis possible,
    and the segment order here is `segments(split)`, the same order `SplitData` uses, so
    baseline and network dumps index segments identically.
    """
    channels = MIC_SUBSETS[subset]
    out = {k: [] for k in ("peak_db", "contrast_db", "el_idx", "az_idx",
                           "az", "el", "range_m", "has_drone", "seg", "frame")}
    seg_ids = []
    for s, seg in enumerate(segments(split)):
        stft, meta = load_cache(seg)
        seg_ids.append(seg.id)
        bf = Beamformer(seg.align, channels, meta["freqs"], device=device,
                        n_subbands=1)
        for i in range(0, len(stft), batch):
            X = np.asarray(stft[i:i + batch][:, channels][:, :, bf.sel])
            maps = bf.maps(torch.from_numpy(X).to(bf.device), log=False)[:, 0]
            flat = maps.flatten(1)
            peak, k = flat.max(1)
            out["peak_db"].append((10 * torch.log10(peak)).cpu().numpy())
            out["contrast_db"].append(
                (10 * torch.log10(peak / flat.mean(1))).cpu().numpy())
            out["el_idx"].append((k // maps.shape[-1]).cpu().numpy())
            out["az_idx"].append((k % maps.shape[-1]).cpu().numpy())
        n = len(meta["has_drone"])
        for k in ("az", "el", "range_m", "has_drone"):
            out[k].append(meta[k])
        out["seg"].append(np.full(n, s))
        out["frame"].append(np.arange(n))
        if verbose:
            print(f"  {seg.id}: {len(stft)} frames")
    r = {k: np.concatenate(v) for k, v in out.items()}
    r["seg_ids"] = seg_ids
    return r


def evaluate(split: str, subset: str = "all24", **kw) -> dict:
    r = run_split(split, subset, **kw)
    return {score: summarize(r[score], r["has_drone"], r["el_idx"], r["az_idx"],
                             r["az"], r["el"], r["range_m"])
            for score in ("peak_db", "contrast_db")}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Classical DAS baseline.")
    p.add_argument("--subsets", nargs="*", default=["all24", "device3", "best6"])
    # test_1 stays sealed until the final numbers, so it is opt-in even here: the baseline
    # has no parameters to overfit, but our choices downstream would be informed by it.
    p.add_argument("--splits", nargs="*", default=["test_2"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    all_results = {}
    for subset in args.subsets:
        for split in args.splits:
            print(f"\n=== {subset} on {split} ===")
            res = evaluate(split, subset, device=args.device)
            for score, m in res.items():
                print(format_summary(f"{score:<12}", m))
            all_results[f"{subset}/{split}"] = res

    path = table_file("baseline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {path}")
