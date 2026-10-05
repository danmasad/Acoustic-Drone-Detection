"""Step 2b gate: is the beamformer right, and do the pinned subsets still match the search?

The interesting check is the third one. Device 4 is exactly coplanar, which makes a
falsifiable prediction from geometry alone: a horizontal array cannot distinguish a source
above its plane from its mirror image below, so its response at elevation -e must equal its
response at +e exactly. A non-coplanar subset must fail that test. Geometry says what the
answer has to be, so it costs nothing and it catches sign errors, transposed axes and
wrong-handed coordinate frames that a plausible-looking energy map would hide.

Run: .venv/bin/python utils/validate/check_arrays.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import arrays                                                  # noqa: E402
from src.config import MIC_SUBSETS, SIDELOBE_DB, segments               # noqa: E402
from src.geometry import coplanar_thickness, directions, mic_positions   # noqa: E402

MIRROR_SRC = (40.0, 28.0)
COPLANAR_TOL = 1e-6      # the mirror ratio must be 1 to numerical precision
DISTINCT = 0.9           # a non-coplanar array must put the mirror clearly below the peak


def mirror_ratio(pos, channels, src=MIRROR_SRC) -> float:
    """Response at (az, -el) divided by response at (az, +el), for a source at (az, +el)."""
    az, el = src
    up, down = directions([az], [el]), directions([az], [-el])
    e_up = arrays.response(pos, channels, src, up)[0]
    e_down = arrays.response(pos, channels, src, down)[0]
    return float(e_down / e_up)


def main() -> int:
    ok = True
    seg = next(s for s in segments("test_1") if s.has_drone)
    pos = mic_positions(seg.align)
    print(f"geometry from {seg.id}\n")
    arrays.report(pos)

    print("\npinned scores against the cached dense search")
    for name, want in SIDELOBE_DB.items():
        got = arrays.sidelobe_db(pos, MIC_SUBSETS[name])["p95"]
        flag = "" if abs(got - want) <= 0.06 else "   FAIL"
        ok &= not flag
        print(f"  {name:<14}{got:>6.2f} dB  config says {want:>5.1f}{flag}")

    print("\nsearched optima against the pinned lists")
    for n, name in ((6, "best6"),):
        got = arrays.best_subset(n)
        same = got == MIC_SUBSETS[name]
        ok &= same
        print(f"  best-{n}: search {got}  config {MIC_SUBSETS[name]}"
              f"{'' if same else '   FAIL'}")
    print(f"  best-4: search {arrays.best_subset(4)}, config keeps "
          f"{MIC_SUBSETS['quad4']} — deliberate: the 4-microphone population spans only "
          f"0.8 dB, so no 'best 4' is meaningful and ours is labelled representative")

    print(f"\nup/down mirror test, source at azimuth {MIRROR_SRC[0]:.0f} deg, "
          f"elevation {MIRROR_SRC[1]:.0f} deg")
    for name in ("device4", "all24", "device3", "best6", "device1"):
        ch = MIC_SUBSETS[name]
        r = mirror_ratio(pos, ch)
        flat = coplanar_thickness(pos[ch])
        if name == "device4":
            bad = abs(r - 1.0) > COPLANAR_TOL
            note = "coplanar: mirror is indistinguishable, as predicted"
        else:
            bad = r > DISTINCT
            note = "resolves the mirror"
        ok &= not bad
        print(f"  {name:<9}flatness {flat:>5.2f} m   mirror/peak {r:>8.5f}   {note}"
              f"{'   FAIL' if bad else ''}")

    print("\nGATE PASS" if ok else "\nGATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
