"""Which microphones to keep, and how badly thinning the array hurts.

Promoted from utils/search_mic_subsets.py and utils/search_best_dense.py, whose dense
search over every non-coplanar 4- and 6-microphone subset is already complete and cached
in results/subset_search.npz. This module is the library form: score a subset, look up a
searched optimum, or re-run the search if the geometry ever changes.

The score is the strongest false peak more than 15 degrees from a single true source,
relative to the true peak, in dB. It is what a brightest-pixel baseline gets wrong, and 0 dB
would mean the array cannot tell the true direction from a phantom at all.
"""
import itertools

import numpy as np

from .config import AZ, C, EL, GRID_SHAPE, MIC_SUBSETS, RESULTS
from .geometry import coplanar_thickness, directions

SEARCH_BAND = np.linspace(200.0, 2200.0, 16)   # the model band, coarsely sampled
EXCLUDE_DEG = 15.0                             # radius counted as "the true peak"
MIN_FLATNESS = 0.15                            # metres; below this, elevation is a mirror


def random_sources(n: int = 250, seed: int = 7) -> np.ndarray:
    """(n, 2) azimuth/elevation pairs. Seed 7 reproduces the completed dense search."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-180, 180, n), rng.uniform(0, 88, n)], axis=1)


def response(pos: np.ndarray, channels, src, targets, band=SEARCH_BAND) -> np.ndarray:
    """Energy the beamformer reports when steered at each target, for one source at src.

    targets is (D, 3) unit vectors, so this also answers questions the 0-90 degree
    elevation grid cannot — such as how much response a horizontal array gives to the
    mirror image of a source below its own plane.
    """
    p = pos[list(channels)]
    d0 = directions([src[0]], [src[1]])[0]
    dtau = (np.asarray(targets) @ p.T - d0 @ p.T) / C                    # (D, M)
    y = np.exp(-2j * np.pi * band[:, None, None] * dtau[None]).sum(-1)   # (F, D)
    return (np.abs(y) ** 2).sum(0)


def beam_pattern(pos: np.ndarray, channels, src, band=SEARCH_BAND) -> np.ndarray:
    """(23, 91) response to one unit-amplitude source at src=(az, el), peak normalized to 1.

    Everything in the map except the peak is an artefact of the array: this is the
    ambiguity a single frame has to live with.
    """
    e = response(pos, channels, src, directions(), band)
    return (e / e.max()).reshape(GRID_SHAPE)


def sidelobe_db(pos: np.ndarray, channels, sources=None, band=SEARCH_BAND) -> dict:
    """Worst false peak over many source directions: max, 95th percentile and median, in dB.

    The 95th percentile is the ranking statistic. A maximum over a few hundred directions is
    dominated by one unlucky draw, which is how the 4-microphone ranking turned out to be
    noise.
    """
    sources = random_sources() if sources is None else np.asarray(sources)
    dirs = directions()
    per_src = np.empty(len(sources), dtype=np.float64)
    for i, src in enumerate(sources):
        e = beam_pattern(pos, channels, src, band).ravel()
        d0 = directions([src[0]], [src[1]])[0]
        far = np.degrees(np.arccos(np.clip(dirs @ d0, -1, 1))) > EXCLUDE_DEG
        per_src[i] = 10 * np.log10(e[far].max() / e.max())
    return {"max": float(per_src.max()), "p95": float(np.percentile(per_src, 95)),
            "median": float(np.median(per_src))}


def load_search() -> dict:
    """The cached dense search: every non-coplanar 4- and 6-subset scored on 250 directions."""
    return dict(np.load(RESULTS / "subset_search.npz", allow_pickle=False))


def best_subset(n: int, exhaustive: bool = False, pos: np.ndarray | None = None,
                sources=None) -> list[int]:
    """Channels of the best n-microphone subset by 95th-percentile false peak.

    Reads the cached search by default. `exhaustive=True` re-enumerates from scratch, which
    took 35 minutes for n=6 and is only needed if the array geometry changes. Note that the
    array has 3-fold symmetry, so a re-run can return a mirror image scoring identically —
    which is exactly why config.MIC_SUBSETS pins channel lists instead of calling this.
    """
    if not exhaustive:
        s = load_search()
        combos, p95 = s[f"combos_{n}"], s[f"p95_{n}"]
        return combos[int(np.argmin(p95))].astype(int).tolist()

    sources = random_sources() if sources is None else sources
    combos = [c for c in itertools.combinations(range(len(pos)), n)
              if coplanar_thickness(pos[list(c)]) >= MIN_FLATNESS]
    scores = [sidelobe_db(pos, c, sources)["p95"] for c in combos]
    return list(combos[int(np.argmin(scores))])


def percentile(n: int, score_db: float) -> float:
    """What fraction of all non-coplanar n-microphone subsets this score beats, in percent."""
    s = load_search()
    return float(100.0 * (s[f"p95_{n}"] < score_db).mean())


def main_lobe_extent(pattern: np.ndarray, drop_db: float = 3.0) -> tuple[float, float]:
    """(elevation span, azimuth span) in degrees of everything within drop_db of the peak.

    Diagnostic for coplanarity: a flat array smears the peak into an elevation stripe
    because directions above and below its plane are indistinguishable.
    """
    j, i = np.where(pattern >= 10 ** (-drop_db / 10))
    return (float(EL[j].max() - EL[j].min()), float(AZ[i].max() - AZ[i].min()))


def report(pos: np.ndarray, names=None) -> None:
    """Print the pinned subsets with their geometry and cached scores."""
    from .geometry import alias_free_hz, spacings

    names = names or list(MIC_SUBSETS)
    print(f"{'subset':<14}{'n':>3}{'apert':>8}{'mingap':>8}{'alias':>9}"
          f"{'flat':>7}{'p95':>8}{'el span':>9}{'az span':>9}")
    for name in names:
        ch = MIC_SUBSETS[name]
        p = pos[ch]
        dmin, dmax = spacings(p)
        pat = beam_pattern(pos, ch, (40.0, 28.0))
        el_span, az_span = main_lobe_extent(pat)
        print(f"{name:<14}{len(ch):>3}{dmax:>7.2f}m{dmin:>7.2f}m{alias_free_hz(p):>7.0f}Hz"
              f"{coplanar_thickness(p):>6.2f}m{sidelobe_db(pos, ch)['p95']:>7.1f}dB"
              f"{el_span:>8.0f}d{az_span:>8.0f}d")
