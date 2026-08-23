"""Detection and localization metrics, computed the same way for the baseline and the CNN.

Deliberately more than the reference paper reports. Their detection number is a
false-positive rate on one five-minute no-drone recording, which says nothing about the
trade-off between missing a drone and crying wolf; precision-recall and false alarms per
hour do.
"""
import numpy as np

from .config import AZ, EL, FRAME_SECONDS, HIT_THRESHOLDS
from .geometry import angular_distance

DISTANCE_BINS = (0, 30, 60, 90, 120, 180, np.inf)


def pr_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and threshold at every distinct score, high threshold first."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    return precision, recall, s


def auprc(scores, labels) -> float:
    """Average precision: the area under the precision-recall curve, computed as the
    step-wise sum so it needs no interpolation choices."""
    precision, recall, _ = pr_curve(scores, labels)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def auroc(scores, labels) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Reported alongside AUPRC because this dataset's test split is 81% positive, which pushes
    AUPRC close to its floor of the prior and makes it a blunt instrument. AUROC does not
    move with the prior, so it is the fairer summary of how separable the two classes are.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(scores), float)
    order = np.argsort(scores)
    s = scores[order]
    # average ranks within ties, so a constant predictor scores exactly 0.5
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def specificity_at_recall(scores, labels, recall: float = 0.95) -> float:
    """Fraction of no-drone frames correctly rejected while catching `recall` of drones."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    if not labels.any() or labels.all():
        return float("nan")
    thr = np.quantile(scores[labels], 1.0 - recall)
    return float((scores[~labels] < thr).mean())


def best_f1(scores, labels) -> dict:
    precision, recall, thr = pr_curve(scores, labels)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1))
    return {"f1": float(f1[k]), "precision": float(precision[k]),
            "recall": float(recall[k]), "threshold": float(thr[k])}


def false_alarms_per_hour(scores, labels, threshold: float) -> float:
    """Rate at which no-drone frames are called drone, per hour of no-drone audio."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    neg = ~labels
    if not neg.any():
        return float("nan")
    hours = neg.sum() * FRAME_SECONDS / 3600.0
    return float((scores[neg] >= threshold).sum() / hours)


def angular_errors(el_idx, az_idx, true_az, true_el) -> np.ndarray:
    """Great-circle error in degrees between the predicted grid cell and the label."""
    return angular_distance(AZ[np.asarray(az_idx)], EL[np.asarray(el_idx)],
                            np.asarray(true_az), np.asarray(true_el))


def axis_errors(el_idx, az_idx, true_az, true_el) -> tuple[np.ndarray, np.ndarray]:
    """Separate azimuth and elevation errors. Thinned arrays resolve elevation far worse
    than azimuth, so the split is informative rather than cosmetic."""
    az_err = np.abs(((AZ[np.asarray(az_idx)] - np.asarray(true_az) + 180) % 360) - 180)
    el_err = np.abs(EL[np.asarray(el_idx)] - np.asarray(true_el))
    return az_err, el_err


def hit_rates(errors, thresholds=HIT_THRESHOLDS) -> dict:
    errors = np.asarray(errors, float)
    return {f"hit_{t}deg": float((errors <= t).mean()) for t in thresholds}


def error_by_distance(errors, range_m, bins=DISTANCE_BINS) -> list[dict]:
    """Median error per distance bin. The reference paper's headline weakness is that
    delay-and-sum degrades with range, so this is the comparison that matters."""
    errors, range_m = np.asarray(errors, float), np.asarray(range_m, float)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (range_m >= lo) & (range_m < hi) & np.isfinite(errors)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "median_deg": float(np.median(errors[m])) if m.any() else float("nan")})
    return out


def summarize(scores, has_drone, el_idx, az_idx, true_az, true_el, range_m) -> dict:
    """Every number one run reports. Localization is scored on drone frames only."""
    has_drone = np.asarray(has_drone).astype(bool)
    out = {"n_frames": int(len(has_drone)), "n_drone": int(has_drone.sum()),
           "auprc": auprc(scores, has_drone), "auroc": auroc(scores, has_drone),
           "specificity_at_95_recall": specificity_at_recall(scores, has_drone)}
    f1 = best_f1(scores, has_drone)
    out |= {f"f1_{k}": v for k, v in f1.items()}
    out["false_alarms_per_hour"] = false_alarms_per_hour(scores, has_drone, f1["threshold"])

    d = has_drone
    err = angular_errors(np.asarray(el_idx)[d], np.asarray(az_idx)[d],
                         np.asarray(true_az)[d], np.asarray(true_el)[d])
    az_err, el_err = axis_errors(np.asarray(el_idx)[d], np.asarray(az_idx)[d],
                                 np.asarray(true_az)[d], np.asarray(true_el)[d])
    out |= {"median_angular_deg": float(np.median(err)),
            "mean_angular_deg": float(err.mean()),
            "median_azimuth_deg": float(np.median(az_err)),
            "median_elevation_deg": float(np.median(el_err))}
    out |= hit_rates(err)
    out["by_distance"] = error_by_distance(err, np.asarray(range_m)[d])
    return out


def format_summary(name: str, m: dict) -> str:
    lines = [f"{name}: {m['n_drone']}/{m['n_frames']} drone frames",
             f"  detection   AUROC {m.get('auroc', float('nan')):.3f}   "
             f"AUPRC {m['auprc']:.3f}   best F1 {m['f1_f1']:.3f} "
             f"(P {m['f1_precision']:.3f} R {m['f1_recall']:.3f})   "
             f"reject {m.get('specificity_at_95_recall', float('nan')) * 100:.0f}% of "
             f"negatives at 95% recall   false alarms {m['false_alarms_per_hour']:.0f}/h",
             f"  direction   median {m['median_angular_deg']:.1f} deg "
             f"(az {m['median_azimuth_deg']:.1f}, el {m['median_elevation_deg']:.1f})   "
             + "  ".join(f"<{t}d {m[f'hit_{t}deg'] * 100:.0f}%" for t in HIT_THRESHOLDS),
             "  by range    " + "  ".join(
                 f"{int(b['lo'])}-{'inf' if not np.isfinite(b['hi']) else int(b['hi'])}m "
                 f"{b['median_deg']:.0f}d(n={b['n']})" for b in m["by_distance"])]
    return "\n".join(lines)
