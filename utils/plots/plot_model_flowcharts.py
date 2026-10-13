"""Flowcharts of the two models: how each is trained, how a new sample becomes a result, and
the layers of each network.

  figures/model/training_flow.png    what calls what, in order, for the localizer and the detector
  figures/model/inference_flow.png   one new 100 ms frame through either model to its output
  figures/model/unet_schema.png      the localizer: U-Net with a mask head and a presence head
  figures/model/detector_schema.png  the per-channel spectral detector
  figures/model/unet_dag.png         the U-Net's computation graph as a DAG
  figures/model/detector_dag.png     the detector's computation graph as a DAG

Layer shapes and parameter counts are traced from `src.models.UNet` and
`src.detector.SpectralDetector`, so the schemas follow the code. The DAGs come from a
`torch.fx` trace of each model: every node is an operation in its forward pass and every edge
a tensor it reads. Each Conv-BatchNorm-ReLU group is one node, and pure reshapes (squeeze,
flatten) and shape lookups are folded into their neighbours.

Run: .venv/bin/python utils/plots/plot_model_flowcharts.py
"""
import operator
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FIGURES, GRID_SHAPE, N_SUBBANDS  # noqa: E402
from src.detector import SpectralDetector  # noqa: E402
from src.models import Conv, UNet  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = FIGURES / "model"
INCH = 0.5                                  # inches per data unit
TITLE_H, LINE_H, PAD = 0.44, 0.34, 0.22     # data units
BODY_PT = 9.0
COLORS = {"entry": "#e6dcf5", "data": "#d9e8f6", "model": "#fde2c4",
          "eval": "#dcefd6", "out": "#ececec"}
LEGEND = {"entry": "entry point", "data": "data and features", "model": "model, loss, update",
          "eval": "evaluation and selection", "out": "files written"}


@dataclass
class Box:
    x: float
    top: float
    w: float
    h: float

    @property
    def bottom(self):
        return self.top - self.h

    @property
    def mid(self):
        return self.top - self.h / 2

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def right(self):
        return self.x + self.w


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w * INCH, h * INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box_height(body: str) -> float:
    return 2 * PAD + TITLE_H + (LINE_H * len(body.splitlines()) if body else 0.0)


def box(ax, x, top, w, title, body="", kind="data", h=None) -> Box:
    b = Box(x, top, w, h or box_height(body))
    ax.add_patch(FancyBboxPatch((b.x, b.bottom), b.w, b.h,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                fc=COLORS[kind], ec="#555555", lw=1.0))
    ax.text(b.cx, top - PAD, title, ha="center", va="top", fontsize=10.5, weight="bold")
    if body:
        ax.text(b.cx, top - PAD - TITLE_H, body, ha="center", va="top", fontsize=BODY_PT,
                linespacing=LINE_H * INCH * 72 / BODY_PT, color="#222222")
    return b


def arrow(ax, p, q, label=None, dashed=False, color="#333333", label_side="right"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=0, shrinkB=0,
                                linestyle="--" if dashed else "-", mutation_scale=14))
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        dx = 0.25 if label_side == "right" else -0.25
        ax.text(mx + dx, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=8.5, style="italic", color=color)


def stack(ax, x, top, w, items, gap=0.55) -> list[Box]:
    """Boxes top to bottom, each joined to the next by an arrow."""
    boxes, y = [], top
    for title, body, kind in items:
        b = box(ax, x, y, w, title, body, kind)
        if boxes:
            arrow(ax, (b.cx, boxes[-1].bottom), (b.cx, b.top))
        boxes.append(b)
        y = b.bottom - gap
    return boxes


def stack_height(items, gap=0.55) -> float:
    return sum(box_height(body) for _, body, _ in items) + gap * (len(items) - 1)


def loop(ax, first: Box, last: Box, side: str, label: str, offset=0.5, color="#1f5fa8"):
    """Return arrow from `last` back up to `first`, drawn outside the column on `side`."""
    edge = first.right if side == "right" else first.x
    xo = edge + offset if side == "right" else edge - offset
    ax.plot([edge, xo, xo], [last.mid, last.mid, first.mid], color=color, lw=1.3)
    arrow(ax, (xo, first.mid), (edge, first.mid), color=color)
    ax.text(xo + (0.12 if side == "right" else -0.12), (first.mid + last.mid) / 2, label,
            rotation=90, ha="left" if side == "right" else "right", va="center",
            fontsize=8.5, style="italic", color=color)


def legend(ax, x, y, kinds=tuple(LEGEND)):
    for k in kinds:
        ax.add_patch(FancyBboxPatch((x, y - 0.3), 0.6, 0.4, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=COLORS[k], ec="#555555", lw=0.8))
        ax.text(x + 0.8, y - 0.1, LEGEND[k], va="center", fontsize=9)
        x += 0.8 + 0.12 * len(LEGEND[k]) + 0.9


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES.parent)}")


# ---------------------------------------------------------------------------- shape tracing

def trace_unet(K: int = 1):
    """Encoder, bottleneck and decoder output shapes (C, H, W), and the parameter count."""
    model = UNet(K * N_SUBBANDS).eval()
    x = torch.zeros(1, K * N_SUBBANDS, *GRID_SHAPE)
    enc, pooled, dec, skips = [], [], [], []
    with torch.no_grad():
        for block in model.down:
            x = block(x)
            enc.append(tuple(x.shape[1:]))
            skips.append(x)
            x = F.max_pool2d(x, 2)
            pooled.append(tuple(x.shape[-2:]))
        x = model.bottleneck(x)
        bottleneck = tuple(x.shape[1:])
        for block, skip in zip(model.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            cin = x.shape[1] + skip.shape[1]
            x = block(torch.cat([x, skip], 1))
            dec.append((cin, *x.shape[1:]))
    hidden = model.presence_head[1]
    return {"enc": enc, "pooled": pooled, "bottleneck": bottleneck, "dec": dec,
            "presence_in": hidden.in_features, "presence_hidden": hidden.out_features,
            "params": model.n_params}


def trace_detector(K: int = 1, F_bins: int = 381):
    model = SpectralDetector(K).eval()
    x = torch.zeros(1, K, F_bins)
    convs = []
    with torch.no_grad():
        for layer in model.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                convs.append((layer.kernel_size[0], layer.stride[0], x.shape[1], x.shape[2]))
    return {"convs": convs, "pooled": model.head[1].in_features,
            "dropout": model.head[0].p, "params": model.n_params}


def hw(shape) -> str:
    return f"{shape[-2]}×{shape[-1]}"


# ---------------------------------------------------------------------------- training

def training_flow():
    w_col, xl, xr, W = 11.0, 1.3, 14.2, 26.5
    shared_body = ("config.segments()  →  stft_cache.SegmentAudio: the 4 recorders aligned to one "
                   "clock\n+ labels.frame_labels: GPS log → azimuth, elevation, has_drone per "
                   "100 ms frame\n→ stft_cache.bandpass (200–4000 Hz)  →  4800-sample frames  →  "
                   "rfft  →  24 mics × 381 bins\n→ cache/<segment>/stft.npy + meta.npz")
    loc = [
        ("utils/train/run_grid.py  →  train.train(subset, K)",
         "config.run_id  →  config.run_dir(run_id)", "entry"),
        ("data.SplitData('train'), SplitData('test_2')",
         "stft_cache.load_cache for every segment\nkeep the subset's mics and bins 200–2200 Hz\n"
         "labels.target_mask: 10° disc on the 23×91 grid\nindex every causal K-frame stack",
         "data"),
        ("train.beamformer_factory  →  beamform.get",
         "Beamformer: geometry.mic_positions + geometry.steering\n"
         "subband_matrix: 16 sub-bands; one tensor per session", "data"),
        ("data.input_stats", "mean and std of the dB maps, 16 training batches", "data"),
        ("models.UNet(K·16), models.Loss('wce'), Adam(5e-3)",
         "mask pos_weight = min(1 / positive-cell rate, 100)\n"
         "presence pos_weight = negatives / positives\nwrite config.json", "model"),
        ("SplitData.batches(32)",
         "shuffled, one segment per batch\nX (B, K, M, F) complex, mask, present", "data"),
        ("data.to_maps(bf, X)",
         "Beamformer.maps  →  dB  →  standardize\n→ (B, K·16, 23, 91)", "data"),
        ("model(x)  →  Loss  →  backward  →  Adam.step",
         "mask logits (B, 23, 91) and presence logit (B)\n"
         "weighted BCE on the mask + weighted BCE on presence", "model"),
        ("train.evaluate(test_2)",
         "same maps, no gradient; brightest cell per frame\n"
         "metrics.summarize: hit_10deg, AUROC, median error", "eval"),
        ("early stopping on hit_10deg",
         "new best only if it gains 0.001: keep its weights\n"
         "stop after 10 epochs without one (min 15, max 60)", "eval"),
        ("restore the best weights",
         "best.pt; final train.evaluate(test_2)\nK > 1: evaluate again with train.reorder "
         "freeze / shuffle / reverse\n→ metrics.json", "out"),
        ("predict.cnn_predictions  →  results/preds/",
         "per-frame scores, centroid and argmax directions\n"
         "predict.verify: recompute metrics.json from the file", "out"),
    ]
    det = [
        ("utils/analysis/detector_loso.py, utils/train/demand_trained.py",
         "→  detector.train(K, ...)", "entry"),
        ("data.SplitData('train'), SplitData('test_2'), all 24 mics",
         "stft_cache.load_cache, bins 200–4000 Hz (381); masks unused\n"
         "split_for_early_stopping: 30 s blocks of the training\nscenes for monitoring "
         "(leave-one-scene-out runs)", "data"),
        ("optional sources",
         "log_filterbank: 64 log-spaced bins\nesc50.Confusers: folds 1–4 to mix in, fold 5 "
         "to test\nambient_pool  →  demand.Ambient, level-matched to our ambient", "data"),
        ("input stats, SpectralDetector(K), Adam(1e-3)",
         "mean and std of dB magnitude, one 64-stack batch\n"
         "pos_weight = negatives / positives (ambient included)", "model"),
        ("SplitData.batches(32)",
         "Confusers.mix_into: ESC-50 added at −5 to +20 dB\n(when training with confusers)",
         "data"),
        ("detector.spectra(X)",
         "|X|  →  filterbank  →  dB  →  standardize\nchannels become samples: (B·24, K, F)\n"
         "each channel takes its frame's label; DEMAND frames added as 0", "data"),
        ("model(x)  →  weighted BCE  →  backward  →  Adam.step",
         "one logit per channel-frame", "model"),
        ("detector.evaluate(monitor set)",
         "frame score = mean of its 24 channel logits\nmetrics.auroc", "eval"),
        ("early stopping on AUROC",
         "new best only if it gains 0.001\nstop after 4 epochs without one (max 12)", "eval"),
        ("last-epoch evaluate, then restore the best",
         "best.pt; final detector.evaluate(test_2)\n"
         "confuser_rejection on ESC-50 fold 5 at 95% recall\n→ metrics.json", "out"),
    ]
    shared_h = box_height(shared_body)
    head = 1.6 + shared_h + 1.4
    H = head + max(stack_height(loc), stack_height(det)) + 1.6
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.55, "Training: what calls what, in order", ha="center", va="top",
            fontsize=15, weight="bold")
    shared = box(ax, xl, H - 1.6, xr + w_col - xl,
                 "once, before any training:  python -m src.stft_cache", shared_body, "entry")
    top = shared.bottom - 1.4
    for x, title in ((xl, "Localizer: U-Net, direction + presence  (src/train.py)"),
                     (xr, "Detector: per-channel drone / no drone  (src/detector.py)")):
        ax.text(x + w_col / 2, top + 0.75, title, ha="center", va="center", fontsize=12,
                weight="bold")
        arrow(ax, (x + w_col / 2, shared.bottom), (x + w_col / 2, top + 1.05))
    a = stack(ax, xl, top, w_col, loc)
    b = stack(ax, xr, top, w_col, det)
    loop(ax, a[5], a[7], "left", "next batch", 0.45)
    loop(ax, a[5], a[9], "right", "next epoch", 0.45)
    loop(ax, b[4], b[6], "left", "next batch", 0.45)
    loop(ax, b[4], b[8], "right", "next epoch", 0.45)
    legend(ax, xl, 0.9)
    save(fig, "training_flow.png")


# ---------------------------------------------------------------------------- inference

def inference_flow():
    W, w_col, xl, xr = 26.5, 11.0, 1.3, 14.2
    shared = [
        ("new sample: 100 ms of 24-channel audio",
         "4800 samples × 24 mics at 48 kHz, plus the K − 1 frames before it", "entry"),
        ("stft_cache.bandpass  →  numpy.fft.rfft",
         "same filter and framing as the cache; keep 200–4000 Hz\n→ (K, 24, 381) complex",
         "data"),
    ]
    loc = [
        ("select the run's mics and 200–2200 Hz", "(K, M, 201) complex", "data"),
        ("Beamformer.maps  (that session's steering)",
         "steer at 23×91 directions, 16 sub-bands, dB\n→ (K, 16, 23, 91)", "data"),
        ("standardize and stack: data.to_maps",
         "input_mean, input_std from config.json\n→ (K·16, 23, 91), oldest frame first",
         "data"),
        ("models.UNet  (best.pt)", "one forward pass", "model"),
    ]
    det = [
        ("detector.spectra", "|X|  →  dB  →  standardize with config\n"
         "→ (24, K, 381): one sample per microphone", "data"),
        ("SpectralDetector  (best.pt)", "one logit per microphone", "model"),
        ("mean of the 24 logits  ≥  threshold",
         "threshold set at 95% recall on test 2", "eval"),
        ("drone / no drone", "", "out"),
    ]
    tops = 1.6
    H = tops + stack_height(shared) + 1.6 + max(stack_height(loc) + 0.55 + 2.4,
                                                 stack_height(det)) + 1.4
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.55, "From a new sample to a result", ha="center", va="top",
            fontsize=15, weight="bold")
    sw = 14.0
    s = stack(ax, (W - sw) / 2, H - tops, sw, shared)
    top = s[-1].bottom - 1.6
    for x, title in ((xl, "Localizer  (predict.cnn_predictions)"),
                     (xr, "Detector  (detector.load_run)")):
        ax.text(x + w_col / 2, top + 0.6, title, ha="center", va="center", fontsize=12,
                weight="bold")
        arrow(ax, (s[-1].cx, s[-1].bottom), (x + w_col / 2, top + 0.9))
    a = stack(ax, xl, top, w_col, loc)
    stack(ax, xr, top, w_col, det)
    hw_ = (w_col - 0.6) / 2
    y = a[-1].bottom - 0.9
    m = box(ax, xl, y, hw_, "mask logits  23×91",
            "models.decode_mask (centroid)\n→ azimuth, elevation", "out")
    p = box(ax, xl + hw_ + 0.6, y, hw_, "presence logit",
            "≥ threshold\n→ drone / no drone", "out")
    for t in (m, p):
        arrow(ax, (t.cx, a[-1].bottom), (t.cx, t.top))
    ax.text(W / 2, 0.6, "The localizer always returns a direction, drone or not; the presence "
            "logit, or the detector, decides whether to trust it.", ha="center", va="center",
            fontsize=10, style="italic")
    save(fig, "inference_flow.png")


# ---------------------------------------------------------------------------- networks

def unet_schema(K: int = 1):
    t, t9 = trace_unet(K), trace_unet(9)
    W, H, bw = 26.0, 20.0, 7.6
    xe, xd, xb = 1.0, W - 1.0 - bw, (W - bw) / 2
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.5, "Localizer: U-Net with a mask head and a presence head",
            ha="center", va="top", fontsize=15, weight="bold")
    conv = "2 × (Conv 3×3  →  BatchNorm  →  ReLU)"
    inp = box(ax, xe, H - 1.7, bw, "input maps",
              f"K frames × {N_SUBBANDS} sub-bands = K·{N_SUBBANDS} channels\n"
              f"{hw(GRID_SHAPE)} (elevation × azimuth), dB, standardized", "data")
    tops = [H - 4.6, H - 8.0, H - 11.4]
    bh = box_height("concatenate\nconv\nshape")
    enc = []
    for i, (shape, top) in enumerate(zip(t["enc"], tops)):
        enc.append(box(ax, xe, top, bw, f"encoder block {i + 1}",
                       f"{conv}\n{shape[0]} channels × {hw(shape)}", "model", h=bh))
    arrow(ax, (inp.cx, inp.bottom), (enc[0].cx, enc[0].top))
    for i in range(2):
        arrow(ax, (enc[i].cx, enc[i].bottom), (enc[i + 1].cx, enc[i + 1].top),
              f"max-pool 2×2  →  {hw(t['pooled'][i])}")
    bn = box(ax, xb, H - 14.6, bw, "bottleneck",
             f"{conv}\n{t['bottleneck'][0]} channels × {hw(t['bottleneck'])}", "model")
    arrow(ax, (enc[2].cx, enc[2].bottom), (bn.x, bn.mid), f"max-pool 2×2  →  "
          f"{hw(t['pooled'][2])}", label_side="left")
    dec = {}
    for i, (cin, c, h, w) in enumerate(t["dec"]):
        level = 2 - i
        dec[level] = box(ax, xd, tops[level], bw, f"decoder block {level + 1}",
                         f"concatenate with the skip  →  {cin} channels\n{conv}\n"
                         f"{c} channels × {h}×{w}", "model", h=bh)
    arrow(ax, (bn.right, bn.mid), (dec[2].cx, dec[2].bottom), "upsample (nearest)")
    for level in (2, 1):
        arrow(ax, (dec[level].cx, dec[level].top), (dec[level - 1].cx, dec[level - 1].bottom),
              "upsample (nearest)")
    for level in range(3):
        y = enc[level].mid
        arrow(ax, (enc[level].right, y), (dec[level].x, y), dashed=True, color="#777777")
        ax.text(W / 2, y + 0.2, "skip connection (copy)", ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#777777")
    out = box(ax, xd, H - 1.7, bw, "mask head",
              "Dropout2d 0.1  →  Conv 1×1  →  1 channel\n"
              f"mask logits {hw(GRID_SHAPE)}  →  decode_mask  →  az, el", "out")
    arrow(ax, (dec[0].cx, dec[0].top), (out.cx, out.bottom))
    ph = box(ax, 4.0, H - 16.9, W - 8.0, "presence head  →  one presence logit (drone / no drone)",
             f"mean-pool ‖ max-pool over {hw(t['bottleneck'])}  →  {t['presence_in']}  →  "
             f"Linear {t['presence_hidden']}  →  LeakyReLU(0.01)  →  Dropout 0.1  →  Linear 1",
             "out")
    arrow(ax, (bn.cx, bn.bottom), (bn.cx, ph.top))
    ax.text(W / 2, 1.3,
            "Padding: circular in azimuth, zero in elevation (the polar 46×46 disc pads zero on "
            f"both).   Parameters: {t['params'] / 1e6:.2f} M at K = {K}, "
            f"{t9['params'] / 1e6:.2f} M at K = 9.\nLoss: weighted BCE on the mask "
            "+ weighted BCE on presence, equal weight.", ha="center", va="center", fontsize=10,
            linespacing=1.5)
    save(fig, "unet_schema.png")


def detector_schema(K: int = 1):
    t = trace_detector(K)
    W, H, bw, gap = 26.4, 8.0, 5.6, 0.8
    xs = [0.8 + i * (bw + gap) for i in range(4)]
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.5, "Detector: per-channel spectral classifier", ha="center",
            va="top", fontsize=15, weight="bold")
    convs = [(f"Conv1d {k}, stride {s}", f"→ BatchNorm → ReLU\n{c} channels × {n} bins")
             for k, s, c, n in t["convs"]]
    row1 = [("input: one microphone", "(K, 381) dB magnitude, 200–4000 Hz\n"
             "standardized; 24 mics = 24 samples", "data")] + \
           [(a, b, "model") for a, b in convs[:3]]
    row2 = [(*convs[3], "model"),
            ("mean ‖ max over frequency", f"{t['pooled']} values", "model"),
            (f"Dropout {t['dropout']}  →  Linear 1", "one logit per microphone", "model"),
            ("frame decision", "mean of the 24 logits ≥ threshold\n→ drone / no drone", "out")]
    h = max(box_height(b) for _, b, _ in row1 + row2)
    top1, top2 = H - 1.8, H - 1.8 - h - 1.4
    r1 = [box(ax, x, top1, bw, *item, h=h) for x, item in zip(xs, row1)]
    r2 = [box(ax, x, top2, bw, *item, h=h) for x, item in zip(reversed(xs), row2)]
    for a, b in zip(r1, r1[1:]):
        arrow(ax, (a.right, a.mid), (b.x, b.mid))
    arrow(ax, (r1[-1].cx, r1[-1].bottom), (r2[0].cx, r2[0].top))
    for a, b in zip(r2, r2[1:]):
        arrow(ax, (a.x, a.mid), (b.right, b.mid))
    ax.text(W / 2, 1.0, f"{t['params'] / 1e3:.1f}k parameters.  Trained with weighted BCE; "
            "every channel takes its frame's label.  Optional 64-bin log filterbank "
            "before the dB step.", ha="center", va="center", fontsize=10)
    save(fig, "detector_schema.png")


# ---------------------------------------------------------------------------- DAGs

DAG_COLORS = {"io": "#ececec", "conv": "#fde2c4", "resample": "#d9e8f6",
              "merge": "#e6dcf5", "dense": "#dcefd6"}
DAG_LEGEND = {"io": "input / output", "conv": "convolution block", "resample":
              "pooling / upsampling", "merge": "concatenate", "dense": "dense head"}
NODE_W, NODE_H = 3.7, 0.8


def traced(model: nn.Module, x: torch.Tensor, leaf=()) -> fx.GraphModule:
    """The model's forward pass as an fx graph, with output shapes on every node."""
    class Tracer(fx.Tracer):
        def is_leaf_module(self, m, name):
            return isinstance(m, leaf) or super().is_leaf_module(m, name)
    gm = fx.GraphModule(model, Tracer().trace(model))
    ShapeProp(gm).propagate(x)
    return gm


def dims(n: fx.Node) -> str:
    """Output shape without the batch axis, e.g. 32×23×91."""
    return "×".join(str(d) for d in n.meta["tensor_meta"].shape[1:])


def dag_edges(gm: fx.GraphModule, rep) -> list[tuple[str, str]]:
    """(source, target) between drawn nodes. `rep(node)` names the node a graph node is drawn
    as, or None to drop it; nodes folded into one drawn node add no edge between themselves."""
    edges = []
    for n in gm.graph.nodes:
        r = rep(n)
        if r is None or n.op == "output":
            continue
        for a in n.all_input_nodes:
            ra = rep(a)
            if ra is not None and ra != r and (ra, r) not in edges:
                edges.append((ra, r))
    return edges


def outputs(gm: fx.GraphModule) -> list[fx.Node]:
    out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    return list(out) if isinstance(out, (tuple, list)) else [out]


def draw_dag(ax, nodes: dict, edges, skip=frozenset(), dotted=frozenset(), node_w=NODE_W):
    """nodes: name -> (x, y, title, subtitle, kind), centred at (x, y)."""
    for name, (x, y, title, sub, kind) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - node_w / 2, y - NODE_H / 2), node_w, NODE_H,
                                    boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=DAG_COLORS[kind], ec="#555555", lw=1.0,
                                    linestyle=":" if name in dotted else "-"))
        ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center", fontsize=9.5,
                weight="bold")
        if sub:
            ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.5, color="#333333")
    for s, t in edges:
        (x0, y0), (x1, y1) = nodes[s][:2], nodes[t][:2]
        if abs(y1 - y0) >= abs(x1 - x0) * (NODE_H / node_w):
            sy = 1 if y1 > y0 else -1
            p, q = (x0, y0 + sy * NODE_H / 2), (x1, y1 - sy * NODE_H / 2)
        else:
            sx = 1 if x1 > x0 else -1
            p, q = (x0 + sx * node_w / 2, y0), (x1 - sx * node_w / 2, y1)
        is_skip = (s, t) in skip
        arrow(ax, p, q, dashed=is_skip or (s, t) in dotted, color="#888888" if is_skip
              else "#333333")
        if is_skip:
            ax.text((p[0] + q[0]) / 2, p[1] + 0.12, f"skip: {nodes[s][3]}", ha="center",
                    va="bottom", fontsize=8, style="italic", color="#777777")


def dag_legend(ax, x, y):
    for k, label in DAG_LEGEND.items():
        ax.add_patch(FancyBboxPatch((x, y - 0.2), 0.5, 0.35,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=DAG_COLORS[k], ec="#555555", lw=0.8))
        ax.text(x + 0.65, y - 0.02, label, va="center", fontsize=8.5)
        x += 0.65 + 0.115 * len(label) + 0.6


def unet_dag(K: int = 1):
    model = UNet(K * N_SUBBANDS).eval()
    gm = traced(model, torch.zeros(1, K * N_SUBBANDS, *GRID_SHAPE), leaf=(Conv,))
    by_name = {n.name: n for n in gm.graph.nodes}

    def rep(n):
        if n.op == "call_function" and n.target in (getattr, operator.getitem):
            return None
        reshape = (n.op == "call_method" and n.target == "squeeze") or (
            n.op == "call_module" and isinstance(gm.get_submodule(n.target), nn.Flatten))
        return rep(n.all_input_nodes[0]) if reshape else n.name

    def label(n):
        if n.op == "placeholder":
            return f"input  (K = {K})", dims(n), "io"
        if n.op == "call_module":
            m = gm.get_submodule(n.target)
            if isinstance(m, Conv):
                return "Conv 3×3 · BN · ReLU", dims(n), "conv"
            if isinstance(m, nn.Linear):
                return f"Linear {m.in_features} → {m.out_features}", "", "dense"
            if isinstance(m, nn.LeakyReLU):
                return f"LeakyReLU {m.negative_slope}", "", "dense"
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                return f"{type(m).__name__} {m.p}", "", "dense"
            if isinstance(m, nn.Conv2d):
                return "Conv 1×1", dims(n), "conv"
        if n.name.startswith("max_pool2d"):
            return "MaxPool 2×2", dims(n), "resample"
        if n.name.startswith("interpolate"):
            return "Upsample (nearest)", dims(n), "resample"
        if n.name.startswith("adaptive_avg"):
            return "global mean pool", "256", "resample"
        if n.name.startswith("adaptive_max"):
            return "global max pool", "256", "resample"
        if n.name.startswith("cat"):
            return "Concatenate", dims(n), "merge"
        raise KeyError(n.name)

    # Rows: the encoder descends the left lane, the decoder climbs the right lane with each
    # Concatenate level with the skip it reads, and the presence head hangs off the bottleneck.
    W, U, xl, xm = 21.0, 1.45, 2.6, 10.5
    xr = W - xl
    rows = {"x": (xl, 0)}
    for i in range(3):
        for j, nm in enumerate((f"down_{i}_a", f"down_{i}_b",
                                "max_pool2d" + (f"_{i}" if i else ""))):
            rows[nm] = (xl, 1 + 3 * i + j)
    rows |= {"bottleneck_a": (xm - 2.3, 10.4), "bottleneck_b": (xm + 2.3, 10.4)}
    right = ["interpolate", "cat_1", "up_0_a", "up_0_b", "interpolate_1", "cat_2", "up_1_a",
             "up_1_b", "interpolate_2", "cat_3", "up_2_a", "up_2_b", "drop", "mask_head",
             "out_mask"]
    for k, nm in enumerate(right):
        rows[nm] = (xr, 9.0 - 1.0 * (k >= 1) - 0.75 * max(k - 1, 0))
    rows |= {"adaptive_avg_pool2d": (xm - 2.3, 11.8), "adaptive_max_pool2d": (xm + 2.3, 11.8),
             "cat": (xm, 12.7)}
    for k, nm in enumerate(("presence_head_1", "presence_head_2", "presence_head_3",
                            "presence_head_4", "out_presence")):
        rows[nm] = (xm, 13.55 + 0.85 * k)

    edges = dag_edges(gm, rep)
    mask_out, presence_out = (rep(n) for n in outputs(gm))
    edges += [(mask_out, "out_mask"), (presence_out, "out_presence")]
    drawn = {e for edge in edges for e in edge}
    if drawn != set(rows):
        raise RuntimeError(f"layout and traced graph disagree: {drawn ^ set(rows)}")

    lo, hi = min(r for _, r in rows.values()), max(r for _, r in rows.values())
    H = (hi - lo) * U + 4.4
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.45, f"Localizer U-Net as a DAG  ({model.n_params / 1e6:.2f} M "
            f"parameters at K = {K})", ha="center", va="top", fontsize=14, weight="bold")
    y_of = lambda r: H - 2.0 - (r - lo) * U                               # noqa: E731
    nodes = {}
    for nm, (x, r) in rows.items():
        if nm == "out_mask":
            nodes[nm] = (x, y_of(r), "mask logits", hw(GRID_SHAPE), "io")
        elif nm == "out_presence":
            nodes[nm] = (x, y_of(r), "presence logit", "drone / no drone", "io")
        else:
            nodes[nm] = (x, y_of(r), *label(by_name[nm]))
    skips = {(s, t) for s, t in edges if s.startswith("down_") and t.startswith("cat_")}
    draw_dag(ax, nodes, edges, skip=skips)
    ax.text(xl, y_of(0) + 0.75, "encoder", ha="center", fontsize=11, style="italic")
    ax.text(xr, y_of(lo) + 0.95, "decoder", ha="center", fontsize=11, style="italic")
    ax.text(xm, y_of(10.4) + 0.75, "bottleneck", ha="center", fontsize=11, style="italic")
    ax.text(xm - NODE_W / 2 - 0.3, y_of(14.8), "presence head", ha="right", fontsize=11,
            style="italic")
    ax.text(W / 2, 1.05, f"Shapes are channels × elevation × azimuth for K = {K} "
            f"({K * N_SUBBANDS} input channels; K frames give K·{N_SUBBANDS}). Convolutions "
            "pad circularly in azimuth.", ha="center", fontsize=9)
    dag_legend(ax, 2.0, 0.45)
    save(fig, "unet_dag.png")


def detector_dag(K: int = 1, F_bins: int = 381):
    model = SpectralDetector(K).eval()
    gm = traced(model, torch.zeros(2, K, F_bins))
    by_name = {n.name: n for n in gm.graph.nodes}
    convs = [i for i, m in enumerate(model.net) if isinstance(m, nn.Conv1d)]

    def rep(n):
        if n.op == "call_method" and n.target == "squeeze":
            return rep(n.all_input_nodes[0])
        if n.name.startswith("net_"):
            i = int(n.name.split("_")[1])
            return f"net_{max(c for c in convs if c <= i)}"
        return n.name

    def label(nm):
        n = by_name[nm]
        if n.op == "placeholder":
            return f"input  (K = {K})", f"{K}×{F_bins} dB, one microphone", "io"
        if nm.startswith("net_"):
            c = model.net[int(nm.split("_")[1])]
            last = by_name[f"net_{int(nm.split('_')[1]) + 2}"]
            return (f"Conv1d {c.kernel_size[0]}, stride {c.stride[0]} · BN · ReLU", dims(last),
                    "conv")
        if nm == "mean":
            return "mean over frequency", dims(n), "resample"
        if nm == "amax":
            return "max over frequency", dims(n), "resample"
        if nm == "cat":
            return "Concatenate", dims(n), "merge"
        m = gm.get_submodule(n.target)
        if isinstance(m, nn.Dropout):
            return f"Dropout {m.p}", "", "dense"
        return f"Linear {m.in_features} → {m.out_features}", "", "dense"

    edges = dag_edges(gm, rep)
    edges += [(rep(outputs(gm)[0]), "out"), ("out", "avg24"), ("avg24", "frame")]
    order = ["x", *(f"net_{c}" for c in convs)]
    W, U, xm, nw = 14.0, 1.15, 7.0, 5.6
    rows = {nm: (xm, k) for k, nm in enumerate(order)}
    r = len(order)
    rows |= {"mean": (xm - 3.0, r), "amax": (xm + 3.0, r)}
    for k, nm in enumerate(("cat", "head_0", "head_1", "out", "avg24", "frame")):
        rows[nm] = (xm, r + 1 + k)
    drawn = {e for edge in edges for e in edge}
    if drawn != set(rows):
        raise RuntimeError(f"layout and traced graph disagree: {drawn ^ set(rows)}")

    H = max(r for _, r in rows.values()) * U + 4.6
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.45, f"Detector as a DAG  ({model.n_params / 1e3:.1f}k parameters)",
            ha="center", va="top", fontsize=13, weight="bold")
    y_of = lambda r: H - 1.9 - r * U                                      # noqa: E731
    extra = {"out": ("logit", "one per microphone", "io"),
             "avg24": ("mean over 24 microphones", "outside the network", "dense"),
             "frame": ("frame score ≥ threshold", "drone / no drone", "io")}
    nodes = {nm: (x, y_of(r), *(extra[nm] if nm in extra else label(nm)))
             for nm, (x, r) in rows.items()}
    draw_dag(ax, nodes, edges, dotted={"avg24", ("out", "avg24")}, node_w=nw)
    ax.text(W / 2, 1.05, "Shapes are channels × frequency bins for one microphone; the 24 "
            "microphones\nof a frame run as 24 separate samples.", ha="center", fontsize=9,
            linespacing=1.4)
    save(fig, "detector_dag.png")


if __name__ == "__main__":
    training_flow()
    inference_flow()
    unet_schema()
    detector_schema()
    unet_dag()
    detector_dag()
