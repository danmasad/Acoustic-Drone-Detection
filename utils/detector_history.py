"""Does the detector get the same benefit from history that the map model does?

The map model's gain from temporal context is the one solid positive result in the project: nine
frames of real history are worth +5.1 pp of hit rate on the six-microphone array, with a 95%
interval of [3.6, 6.8] that survives correction for the whole family of claims we are making.

The detector never got the chance. It has always been K=1 — one frame, one channel, 381 bins — and
it is also our worst result: 0.66 AUROC, and 24,881 false alarms per hour on real ambient at the
threshold that catches 95% of drones. Post-hoc smoothing of its scores helped, which is indirect
evidence that the information is there and unused, but smoothing a score is not the same as
letting a model learn across frames.

There is a mechanism to expect here, and it is not the same one as the map model's. A drone is a
sustained harmonic stack that persists for minutes; most of what makes this detector fire is
transient — a door, a bird, a passing car. That difference does not exist in a single frame at
all. If half a second of context lets it separate sustained from transient, the false-alarm rate
should fall much faster than the AUROC rises, so both are reported and the false-alarm rate is
the one to read.

Compares K=1, 5 and 9 at matched width and seed. AUROC comes from `metrics.json`, false alarms
from the DEMAND evaluation, which has to have been run separately for each detector.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, RUNS                              # noqa: E402

DEPTHS = (1, 5, 9)
SEEDS = (0, 1, 2)


def rows(K: int) -> list[dict]:
    out = []
    for s in SEEDS:
        run_id = f"detector_K{K}_w32_s{s}_full"
        p = RUNS / run_id / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        fa = RESULTS / f"demand_false_alarms_{run_id}.json"
        d = {"run_id": run_id, "seed": s, "auroc": m["test_2"]["auroc"],
             "auprc": m["test_2"]["auprc"],
             "reject_at_95": m["test_2"]["specificity_at_95_recall"]}
        if fa.exists():
            f = json.loads(fa.read_text())
            d["demand_per_hour"] = f["pooled_per_hour"]
            d["demand_median_scene"] = f.get("median_scene_per_hour")
        if "confuser_rejection" in m and m["confuser_rejection"]:
            d["esc50_fire_rate"] = m["confuser_rejection"].get("fire_rate")
        out.append(d)
    return out


def col(rs: list[dict], key: str):
    v = [r[key] for r in rs if key in r and r[key] is not None]
    return np.array(v, float) if v else None


def show(name: str, rs: list[dict], key: str, scale: float = 1.0, fmt: str = "{:.3f}") -> str:
    v = col(rs, key)
    if v is None:
        return "-"
    s = fmt.format(v.mean() * scale)
    return s + (f" +/- {fmt.format(v.std(ddof=1) * scale)}" if len(v) > 1 else "")


def main() -> int:
    print(__doc__.split("\n\n")[0])
    data = {K: rows(K) for K in DEPTHS}
    have = {K: r for K, r in data.items() if r}
    if 1 not in have:
        sys.exit("no K=1 detector to compare against")

    print(f"\n{'depth':<8}{'seeds':>7}{'AUROC':>16}{'AUPRC':>16}"
          f"{'reject@95 recall':>20}{'DEMAND alarms/h':>20}")
    for K, rs in have.items():
        print(f"K={K:<6}{len(rs):>7}{show(name='', rs=rs, key='auroc'):>16}"
              f"{show('', rs, 'auprc'):>16}"
              f"{show('', rs, 'reject_at_95', 100, '{:.0f}') + '%':>20}"
              f"{show('', rs, 'demand_per_hour', 1, '{:,.0f}'):>20}")

    out = {"depths": {str(K): rs for K, rs in have.items()}}
    base = have[1]
    for K, rs in have.items():
        if K == 1 or len(rs) < 2 or len(base) < 2:
            continue
        a, b = col(base, "auroc"), col(rs, "auroc")
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"\nK=1 against K={K}, AUROC: {b.mean() - a.mean():+.3f} "
              f"(Welch t {t:.2f}, p {p:.4f})")
        out[f"auroc_k1_vs_k{K}"] = {"difference": float(b.mean() - a.mean()),
                                    "welch_t": float(t), "welch_p": float(p)}
        fa_a, fa_b = col(base, "demand_per_hour"), col(rs, "demand_per_hour")
        if fa_a is not None and fa_b is not None:
            print(f"K=1 against K={K}, DEMAND false alarms: {fa_a.mean():,.0f}/h to "
                  f"{fa_b.mean():,.0f}/h ({100 * (fa_b.mean() / fa_a.mean() - 1):+.0f}%)")
            out[f"demand_k1_vs_k{K}"] = {"from": float(fa_a.mean()), "to": float(fa_b.mean())}

    # The comparison that decides what this means: the map model's own history gain, measured by
    # the freezing intervention on the same test split.
    sig = RESULTS / "significance.json"
    if sig.exists():
        s = json.loads(sig.read_text()).get("freeze_device3_k9_pooled", {})
        if s:
            print(f"\nFor scale, the map model's history gain on the same split: "
                  f"{s['mean']:+.1f} pp of hit rate, 95% CI "
                  f"[{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}].")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detector_history.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {RESULTS / 'detector_history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
