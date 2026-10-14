"""Build the talk (docs/presentation.pptx) and the two figures in the report.

Numbers are the ones in docs/report.md (12 October 2026). The headline bars are
checked against runs/grid/*/metrics.json before anything is written.

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/build_slides.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
DOCS = ROOT / "docs"
OUT = DOCS / "presentation.pptx"

FONT = "Helvetica Neue"
BG, DARK = "FAFAF9", "0A0A0A"
INK, BODY, MUTED, LINE, CARD = "1C1917", "44403C", "78716C", "E7E5E4", "FFFFFF"
BLUE, GREEN, RED, AMBER, GREY = "2563EB", "16A34A", "DC2626", "D97706", "A8A29E"
SW, SH = 13.333, 7.5
TOTAL = 14

# Seed means on the matched site, before the filter. Delay-and-sum is one run;
# 32.6% rounds to the 33 in the report table.
HEADLINE = [
    ("Delay-and-sum, 24", 32.6, 0.0, GREY),
    ("24 mics, 1 frame", 70.5, 1.5, BLUE),
    ("Device 3, 1 frame", 69.3, 0.7, GREEN),
    ("24 mics, 9 frames", 70.1, 2.7, BLUE),
    ("Device 3, 9 frames", 73.2, 1.4, GREEN),
    ("Searched 6, 1 frame", 52.1, 0.8, AMBER),
]
# Same device-3 K=9 weights; only the depth of real history changes.
HISTORY_GAIN = [0.5, 4.4, 8.7, 21.8, 3.0]
HISTORY_BINS = ["0–30 m", "30–60", "60–90", "90–120", "120–180"]


def rgb(h):
    return RGBColor.from_string(h)


def box(slide, x, y, w, h, fill=CARD, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    if radius is not None:
        s.adjustments[0] = radius
    return s


def card(slide, x, y, w, h, fill=CARD, line=LINE):
    return box(slide, x, y, w, h, fill, line, MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)


def text(slide, x, y, w, h, paras, size=16, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, after=6, line=1.08):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        p.line_spacing = line
        runs = [(para, {})] if isinstance(para, str) else para
        for t, st in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(st.get("size", size))
            f.bold = st.get("bold", bold)
            f.italic = st.get("italic", False)
            f.color.rgb = rgb(st.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=16, dot=BLUE, after=8):
    paras = []
    for it in items:
        runs = [("●  ", {"color": dot, "size": size - 5})]
        runs += [(it, {})] if isinstance(it, str) else it
        paras.append(runs)
    return text(slide, x, y, w, h, paras, size=size, after=after, line=1.08)


def hline(slide, x1, y1, x2, y2, color=LINE, width=0.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                   Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    return c


def base(prs, n, kicker, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    if kicker:
        text(s, 0.55, 0.32, 12, 0.28, kicker.upper(), size=12, color=BLUE, bold=True)
    if title:
        text(s, 0.55, 0.58, 12.2, 0.7, title, size=28, color=INK, bold=True, after=0)
    if n:
        text(s, 0.55, 7.12, 9.2, 0.24, "Hearing drones in 3D  ·  Shai Livshits & Dan Masad", size=11, color=MUTED)
        text(s, 11.3, 7.12, 1.5, 0.24, f"{n}  /  {TOTAL}", size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body.strip()


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def stat(slide, x, y, w, h, big, label, color=INK, big_size=26):
    card(slide, x, y, w, h)
    text(slide, x + 0.18, y + 0.14, w - 0.36, 0.5, big, size=big_size, color=color, bold=True)
    text(slide, x + 0.18, y + 0.68, w - 0.36, h - 0.82, label, size=13, color=BODY, after=0)


def table(slide, x, y, widths, headers, rows, size=14, rh=0.42):
    cx = x
    for j, h in enumerate(headers):
        text(slide, cx, y, widths[j] - 0.08, 0.36, h, size=12, color=MUTED, bold=True,
             align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, after=0)
        cx += widths[j]
    hline(slide, x, y + 0.40, x + sum(widths), y + 0.40, color=MUTED, width=1.0)
    for i, row in enumerate(rows):
        yy = y + 0.50 + i * rh
        cx = x
        for j, cell in enumerate(row):
            text(slide, cx, yy, widths[j] - 0.08, 0.34, cell, size=size, color=INK,
                 bold=(j == 0), align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
            cx += widths[j]
        if i < len(rows) - 1:
            hline(slide, x, yy + rh - 0.06, x + sum(widths), yy + rh - 0.06, width=0.6)
    return y + 0.50 + len(rows) * rh


def _style(ax):
    ax.set_facecolor("#FAFAF9")
    for sp in ax.spines.values():
        sp.set_color("#E7E5E4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#44403C", labelsize=12, length=0)
    ax.grid(axis="x", color="#E7E5E4", zorder=0)


def check_headline() -> None:
    """The bars have to be the runs, not a retyped copy of the report."""
    expected = {("all24", 1): (70.5, 1.5), ("all24", 9): (70.1, 2.7),
                ("device3", 1): (69.3, 0.7), ("device3", 9): (73.2, 1.4),
                ("best6", 1): (52.1, 0.8)}
    for (subset, k), (mean, sd) in expected.items():
        vals = []
        for seed in range(3):
            path = ROOT / "runs" / "grid" / f"{subset}_K{k}_w32_b16_wce_s{seed}" / "metrics.json"
            vals.append(json.loads(path.read_text())["test_2"]["hit_10deg"] * 100)
        got_m, got_s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        if abs(got_m - mean) > 0.15 or abs(got_s - sd) > 0.15:
            raise SystemExit(f"{subset} K={k}: runs say {got_m:.2f} ± {got_s:.2f}, "
                             f"report says {mean} ± {sd}")


def save_charts() -> dict:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "axes.unicode_minus": False,
    })
    check_headline()

    fig, ax = plt.subplots(figsize=(7.4, 3.55), dpi=160)
    fig.patch.set_facecolor("#FAFAF9")
    labels = [r[0] for r in HEADLINE][::-1]
    means = [r[1] for r in HEADLINE][::-1]
    sds = [r[2] for r in HEADLINE][::-1]
    colors = ["#" + r[3] for r in HEADLINE][::-1]
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=sds, color=colors, height=0.62, zorder=2,
            error_kw={"ecolor": "#44403C", "capsize": 3, "capthick": 1, "elinewidth": 1})
    for i, (m, s) in enumerate(zip(means, sds)):
        label = f"{m:.1f}" if s == 0 else f"{m:.1f} ± {s:.1f}"
        ax.text(m + s + 1.3, i, label, va="center", ha="left", color="#1C1917", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of drone frames within 10°")
    _style(ax)
    fig.tight_layout()
    headline = DOCS / "headline_within_10.png"
    fig.savefig(headline, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.15), dpi=160)
    fig.patch.set_facecolor("#FAFAF9")
    xs = np.arange(len(HISTORY_GAIN))
    colors = ["#16A34A" if g >= 8 else "#2563EB" if g >= 2 else "#A8A29E" for g in HISTORY_GAIN]
    ax.bar(xs, HISTORY_GAIN, color=colors, width=0.66, zorder=2)
    for i, g in enumerate(HISTORY_GAIN):
        ax.text(i, g + 0.6, f"+{g:.1f}", ha="center", va="bottom", color="#1C1917", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(HISTORY_BINS)
    ax.set_ylim(0, 28)
    ax.set_ylabel("Gain, points within 10°")
    _style(ax)
    ax.grid(axis="y", color="#E7E5E4", zorder=0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    history = DOCS / "history_gain.png"
    fig.savefig(history, facecolor=fig.get_facecolor())
    plt.close(fig)

    runs = []
    for seed in range(3):
        path = ROOT / "runs" / "grid" / f"device3_K1_w32_b16_wce_s{seed}" / "metrics.json"
        m = json.loads(path.read_text())
        runs.append((m["best_epoch"], m["history"]))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 3.35), dpi=150)
    fig.patch.set_facecolor("#FAFAF9")
    palette = ["#2563EB", "#0F766E", "#D97706"]
    for seed, ((best, hist), color) in enumerate(zip(runs, palette)):
        ep = [h["epoch"] for h in hist]
        ax0.plot(ep, [h["hit_10deg"] * 100 for h in hist], color=color, lw=1.8, label=f"seed {seed}")
        ax0.scatter([best], [hist[best]["hit_10deg"] * 100], s=36, color=color, zorder=3)
        ax1.plot(ep, [h["train_loss"] for h in hist], color=color, lw=1.6)
        ax1.plot(ep, [h["val_loss"] for h in hist], color=color, lw=1.4, ls="--")
    ax0.set_ylim(52, 78)
    ax0.set_ylabel("Within 10° on test 2 (%)")
    ax0.set_title("The direction. This holds.", loc="left", fontsize=13, color="#1C1917")
    ax0.legend(frameon=False, fontsize=11)
    ax1.set_ylabel("Pixel loss")
    ax1.set_title("The loss. This is not the score.", loc="left", fontsize=13, color="#1C1917")
    for ax in (ax0, ax1):
        _style(ax)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#E7E5E4", zorder=0)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=3.0)
    training = DOCS / "_training_device3_k1.png"
    fig.savefig(training, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"headline": headline, "history": history, "training": training}


def s01_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK)
    text(s, 0.7, 1.55, 6.2, 0.3, "DEEP LEARNING 2026B  ·  APPLIED", size=13, color="60A5FA", bold=True)
    text(s, 0.7, 2.05, 6.2, 1.8, ["Acoustic Drone", "Detection"], size=44, color="F5F5F4", bold=True, after=0, line=0.96)
    text(s, 0.7, 4.35, 5.9, 0.32, "Shai Livshits  ·  Dan Masad", size=16, color="F5F5F4")
    text(s, 0.7, 4.78, 5.9, 0.28, "12 October 2026", size=14, color="78716C")
    picture(s, FIG / "flight_hemisphere_mar18_f2.gif", 7.15, 1.7, w=5.4)
    notes(s, """
About fifteen minutes between us, and the last minute is the three limits.
We say why the problem is worth a project, then we move. We do not read the slide.
The clip is one flight on the matched site: GPS against our direction, green inside 10°, red outside.
""")


def s02_problem(prs):
    s = base(prs, 2, "The problem", "A direction, ten times a second")
    bullets(s, 0.55, 1.55, 7.3, 4.6, [
        "A small drone often flies with no radio link. A camera loses it in poor visibility, and a radio detector has nothing to hear.",
        "The rotors are loud, and the sound is a tonal stack. A ground array can report it.",
        "Every tenth of a second: is a drone audible, and in which direction.",
        "Published arrays use 24 or more synchronized microphones.",
    ], size=18, after=14)
    card(s, 8.15, 1.55, 4.6, 4.7, fill=DARK, line=None)
    text(s, 8.4, 1.78, 4.15, 0.55, "The Research Question", size=16, color="60A5FA", bold=True)
    text(s, 8.4, 2.5, 4.15, 2.0,
         "Can a few frames of history let six microphones match twenty-four?",
         size=20, color="F5F5F4", line=1.15)
    text(s, 8.4, 4.7, 4.15, 1.2,
         "Six microphones is one off-the-shelf recorder. No clock shared with a second device. About four times cheaper.",
         size=15, color="A8A29E", line=1.12)
    notes(s, """
We open with the operational constraint, then our question: can a few frames of history let six microphones match twenty-four?
Rodríguez et al., IWAENC 2026, beamform and then segment the map. They released the data and not the training code, and their own future-work sentence is temporal context. We reimplemented their system so the comparison is ours to measure.
We hold the answer. It lands on the results.
""")


def s03_data(prs):
    s = base(prs, 3, "Data", "One drone, two sites, three jobs")
    text(s, 0.55, 1.45, 12.2, 0.4,
         "Western Sydney University array. A DJI Air 3, four recorders of six, 48 kHz, GPS at 10 Hz. Splits are the paper's.",
         size=16, color=BODY)
    table(s, 0.55, 2.05, [2.0, 3.3, 2.2, 4.6],
          ["Split", "Where", "Drone audio", "What we used it for"],
          [
              ["Train", "Site 1, two flights", "52 min", "Fit the weights"],
              ["Test 2", "Site 1, later flight", "20 min", "Early stop: best hit rate"],
              ["Test 1", "A second site", "3 min", "Sealed. Scored once"],
          ], size=16, rh=0.55)
    card(s, 0.55, 4.55, 12.2, 2.2)
    text(s, 0.8, 4.75, 11.7, 0.35, "The weak part is the quiet audio", size=16, color=INK, bold=True)
    text(s, 0.8, 5.2, 11.7, 1.3,
         "The release labels about five minutes with no drone, one recording cut in half between train and test. "
         "We mined the pre-takeoff audio the splits never use, checked by beamformed contrast that it is ambient, and dropped 20 seconds for motor spin-up. "
         "Training negatives went from 4.8 to 8.2 minutes, and from one scene to three. That is still not enough to learn a detector.",
         size=15, color=BODY, line=1.12)
    notes(s, """
"Early stop: best hit rate" means there is no third split. Every epoch we score test 2 by the share of drone frames within 10°, and the weights we save are that best epoch, not the last epoch and not the lowest loss. Test 2 is therefore also the flight in the main table. It is not a pure held-out test.
Test 1 is the one we never used to choose anything. Three minutes, scored once, after the choices were fixed.
The negatives are three scenes. That number comes back when detection fails.
Targets are 10° discs. Row k of the GPS log is frame k. That is the whole alignment.
""")


def s04_model(prs):
    s = base(prs, 4, "Method", "What we tried")
    text(s, 0.55, 1.38, 6, 0.26, "WHERE WE STARTED", size=12, color=BLUE, bold=True)
    basis = [
        ("The map", "Delay-and-sum. Each 100 ms frame becomes 23 by 91 directions, in 16 bands."),
        ("The U-Net", "The paper's network. It reads that map as an image and paints a mask. The direction is read off the mask."),
        ("The baseline", "The brightest cell of the map itself. No learning, so a gain over it is the network."),
    ]
    for i, (title, body) in enumerate(basis):
        x = 0.55 + i * 4.2
        card(s, x, 1.72, 4.02, 2.05)
        text(s, x + 0.16, 1.86, 3.7, 0.34, title, size=16, color=BLUE, bold=True)
        text(s, x + 0.16, 2.28, 3.7, 1.3, body, size=15, color=BODY, line=1.08)
    text(s, 0.55, 3.96, 8, 0.26, "HOW WE TRIED TO GET AN EDGE", size=12, color=INK, bold=True)
    added = [
        ("History", "K past frames stacked as channels. K is 1, 5, or 9."),
        ("Detection, two ways", "A presence score on the U-Net. And an 86k network on one microphone, with no array."),
        ("More quiet audio", "Pre-takeoff from these flights, 4.8 to 8.2 minutes. Other recordings mixed in. 18 soundscapes as a test only."),
        ("A filter", "After the direction. Anything past 10° waits half a second. Not trained."),
    ]
    for i, (title, body) in enumerate(added):
        x = 0.55 + i * 3.2
        card(s, x, 4.3, 3.05, 2.55)
        text(s, x + 0.14, 4.42, 2.76, 0.58, title, size=15, color=INK, bold=True)
        text(s, x + 0.14, 5.05, 2.76, 1.6, body, size=14, color=BODY, line=1.06)
    notes(s, """
The goal is already on the problem slide. This one is the basis, then what we added.
Where we started is the paper. Delay-and-sum builds the map. Their U-Net paints a mask on it, and the direction is read off that mask. The baseline is the brightest cell of the map with no learning, so a gain over it is the network. Their detection number was whether that direction sat still. We did not start from a detector.
The bottom row is what we added. History is K frames as channels. Detection is two models: the presence branch at the bottom of the U-Net, and the 86k network on one microphone so the test does not depend on the array. More quiet audio is three tries: pre-takeoff from these flights, 4.8 to 8.2 minutes and one scene to three; other recordings mixed in as not-a-drone, which raised the average a little and left the different-day fold at chance; DEMAND, 18 soundscapes, as a test only. At a usable threshold the detector still fires on most of them, about 22,000 false alarms an hour. The filter runs after the direction and is not trained. A step inside 10° of the last direction is taken at once. A larger jump waits half a second.
Azimuth uses circular padding. The grid uses 32 filters.
""")


def save_unet(path: Path) -> None:
    """The trained U-Net: depth 3, base 32, mask head plus the presence head."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.0, 5.35), dpi=150)
    fig.patch.set_facecolor("#FAFAF9")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#FAFAF9")

    def box(x, y, w, h, fc, ec, title, sub, tc="#1C1917", sc="#44403C", ts=15, ss=11):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.6, mutation_aspect=0.55, zorder=3))
        ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center",
                fontsize=ts, color=tc, fontweight="medium", zorder=4)
        if sub:
            ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=ss, color=sc, zorder=4)

    def arrow(x1, y1, x2, y2, color="#2563EB", lw=1.5, style="-"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=lw, linestyle=style, zorder=2,
            shrinkA=0, shrinkB=0))

    # Encoder left, decoder right. Sizes are what max-pooling a 23×91 map actually yields.
    xl, xr, w, h = 2.55, 7.55, 2.15, 0.78
    ys = [4.35, 3.15, 1.95]
    enc = [("#DBEAFE", "#2563EB", "32", "23 × 91"),
           ("#DBEAFE", "#2563EB", "64", "11 × 45"),
           ("#DBEAFE", "#2563EB", "128", "5 × 22")]
    dec = [("#DCFCE7", "#16A34A", "32", "23 × 91"),
           ("#DCFCE7", "#16A34A", "64", "11 × 45"),
           ("#DCFCE7", "#16A34A", "128", "5 × 22")]
    for y, (fc, ec, title, sub) in zip(ys, enc):
        box(xl, y, w, h, fc, ec, title, sub)
    for y, (fc, ec, title, sub) in zip(ys, dec):
        box(xr, y, w, h, fc, ec, title, sub)

    box(5.05, 0.72, 2.15, 0.78, "#1D4ED8", "#1D4ED8", "256", "2 × 11", tc="#F8FAFC", sc="#DBEAFE", ss=11)

    box(0.95, 4.35, 1.55, 0.7, "#FFFFFF", "#D6D3D1", "16 × K", "23 × 91", ts=13, ss=10)
    box(9.55, 4.35, 1.7, 0.7, "#FFFFFF", "#16A34A", "Mask", "23 × 91", ts=13, ss=10)
    box(11.85, 4.35, 1.85, 0.7, "#FFFFFF", "#44403C", "Direction", "brightest cell", ts=13, ss=10)

    box(9.15, 0.72, 2.05, 0.78, "#FEF3C7", "#D97706", "Avg + max", "512, then 128", ts=13, ss=10)
    box(11.7, 0.72, 1.9, 0.78, "#FEF3C7", "#D97706", "Presence", "drone in frame?", ts=13, ss=10)

    arrow(1.73, 4.35, xl - w / 2, 4.35)
    arrow(xr + w / 2, 4.35, 8.7, 4.35, color="#16A34A")
    arrow(10.4, 4.35, 10.92, 4.35, color="#44403C")
    for y0, y1 in zip(ys, ys[1:]):
        arrow(xl, y0 - h / 2, xl, y1 + h / 2)
        arrow(xr, y1 + h / 2, xr, y0 - h / 2, color="#16A34A")
    arrow(xl, ys[-1] - h / 2, 5.05, 0.72 + 0.39)
    arrow(5.05, 0.72 + 0.39, xr, ys[-1] - h / 2, color="#16A34A")
    for y in ys:
        arrow(xl + w / 2, y, xr - w / 2, y, color="#93C5FD", lw=1.3)
    arrow(6.13, 0.72, 8.12, 0.72, color="#D97706")
    arrow(10.18, 0.72, 10.75, 0.72, color="#D97706")

    ax.text(xl + 0.28, 3.78, "pool", fontsize=9, color="#2563EB", ha="left")
    ax.text(8.78, 3.75, "nearest + skip", fontsize=9, color="#16A34A", ha="left", va="center")
    ax.text(5.05, 2.55, "skip", fontsize=10, color="#60A5FA", ha="center")
    ax.text(6.5, 5.05, "Each box is two 3×3 convolutions, batch norm, and ReLU. Azimuth wraps, so the left and right edges are one bearing.",
            fontsize=11, color="#44403C", ha="center")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def s05_network(prs):
    s = base(prs, 5, "The network", "One U-Net, two answers")
    picture(s, DOCS / "unet_structure.png", 0.2, 1.35, w=12.9)
    notes(s, """
We walk the U. Down the left, the 23 by 91 map is pooled three times, to 2 by 11. The number in each box is the channel count at 32 base filters. A skip copies the finer map across, because the direction lives in a small patch of sky.
The green side rebuilds the full map. Nearest-neighbor resizes the coarse map to the skip's exact size, the two are concatenated, and the two convolutions mix them. Pooling does not divide the 23-row map by two, so a plain doubling would miss the skip. The mask is one score per cell, and the direction is the brightest cell. That is the result we report.
The amber branch at the bottom is the presence head. It was supposed to say whether a drone is in the frame. Average and max of the bottleneck, then 128 units and a leaky ReLU, then one score. A plain ReLU in that layer died in one seed, which is why it is leaky. The mask does not pass through it, so localization kept training. The score itself did not generalize, which is the detection slide.
K frames of 16 bands enter as channels. Only the first layer grows with K. The rest of the U is the same at one, five, and nine frames. About 2.0 million weights.
""")


def s06_hardware(prs):
    s = base(prs, 6, "The array", "Four recorders, six microphones each")
    picture(s, FIG / "array_geometry.png", 0.28, 1.38, w=8.15)
    points = [
        ("Colour is the recorder", "Purple, blue, teal, yellow. Each is one Zoom recorder, so the six share a clock."),
        ("Teal is device 3", "The six we train. 1.09 m across, the outer microphones of one recorder."),
        ("The bottom row", "Other subsets we drew. Centre is device 3 again. The searched six is not a picture here."),
        ("Then the map", "Next slide is one simulated source through this geometry."),
    ]
    for i, (title, body) in enumerate(points):
        y = 1.48 + i * 1.3
        text(s, 8.6, y, 4.2, 0.32, title, size=16, color=INK, bold=True)
        text(s, 8.6, y + 0.34, 4.2, 0.85, body, size=14, color=BODY, line=1.05)
    notes(s, """
We show the hardware before the simulated response, so the map has something to point at.
Top left: four colours, four recorders. Teal is device 3, one recorder, 1.09 m. The figure's own label says that subset is alias-free only below about 380 Hz, and our maps run to 2.2 kHz, which is why the next slide has phantoms.
The title on the figure says test-1 geometry. It is the same hardware. We are not scoring the sealed site on this slide.
The bottom centre is device 3 again. The searched six spans two recorders and is not one of these panels.
""")


def s06_map(prs):
    s = base(prs, 7, "The map", "A phantom can look like the source")
    picture(s, FIG / "array_beampattern.png", 0.4, 1.45, w=8.15)
    points = [
        ("1.47 kHz", "Spacing is alias-free only below this. The maps run to 2.2 kHz."),
        ("3.6 dB", "Not the number on the panels. Over 250 directions, the phantom all 24 still leave. Nearly half the true peak."),
        ("Device 3", "One recorder. Six outer microphones, 1.09 m across. No shared clock."),
        ("Why a network", "The brightest pixel cannot tell the phantom from the source. The mask sees the whole pattern."),
    ]
    for i, (title, body) in enumerate(points):
        y = 1.48 + i * 1.3
        text(s, 8.75, y, 4.05, 0.32, title, size=16, color=INK, bold=True)
        text(s, 8.75, y + 0.34, 4.05, 0.85, body, size=14, color=BODY, line=1.05)
    notes(s, """
This figure is a simulation, not a recording. We place one source at the cyan cross, apply only the geometric delay at each microphone, and beamform every cell. No noise, no drone spectrum, plane-wave delays. Every bright spot besides the cross is a phantom the geometry makes.
The dB numbers printed on the panels are that one source. All 24 shows about −4.2 dB there. The 3.6 dB callout is a different summary: over 250 directions, the 95th percentile of the strongest phantom, for all 24 and also for the best six. 3 dB down would be half the power. 3.6 dB is still nearly half, so one frame's brightest pixel cannot be trusted. That is the margin the network is asked to beat.
Device 4, the flat ring, is a vertical stripe. It cannot tell elevation from its mirror. That is why some six-microphone sets fail.
All 24 includes that flat recorder and a pair 0.17 m apart, so uniform delay-and-sum gives a bad microphone the same vote as a good one. That is why six can match twenty-four before any learning.
""")


def s08_training(prs, charts):
    s = base(prs, 8, "Training", "We keep the best test-2 hit rate, not the last epoch")
    stats = [
        ("Adam, 5×10⁻³", "Constant. No weight decay. Batch 32. Batch norm, dropout 0.1."),
        ("Weighted loss", "1.4% of pixels are the drone. Unweighted, empty sky wins."),
        ("Patience 10", "At least 15 epochs, at most 60. Three seeds. Test 1 stays sealed."),
    ]
    for i, (big, label) in enumerate(stats):
        stat(s, 0.55 + i * 4.2, 1.45, 4.02, 1.35, big, label, color=BLUE, big_size=22)
    picture(s, charts["training"], 0.5, 3.0, w=12.3)
    text(s, 0.55, 6.48, 12.2, 0.55,
         "Device 3, one frame. The dot is the epoch we keep. Solid is training loss, dashed is test 2. The direction stays near 70%. The loss rises because a few confident wrong pixels dominate it.",
         size=14, color=BODY)
    notes(s, """
The network does work. Delay-and-sum on these frames is 33% within 10°. The left plot sits near 70%, and the dot is the epoch we keep. The last epoch is 3 to 10 points worse on that score, which is why we do not keep it.
The dashed line is the loss on test 2, and it rises. That loss is weighted cross-entropy on every pixel, and about 1.4% of pixels are the drone. A few confident mistakes on background pixels drive it up without moving the decoded direction. Generalization of the direction is the flat left plot. Generalization of detection is a different failure, on the detection slide.
Test 2 is the selection set and the main reported flight. Test 1 was scored once, after the choices were fixed.
The hit rate jitters by several points from epoch to epoch. A one-point gap between two trained models is inside that. The yardstick is the seed spread, ±1.2 points pooled.
No learning-rate schedule. We froze the recipe so the rows stay comparable.
One seed killed a ReLU in the presence head and sat at chance detection while localization trained. The head is a leaky ReLU now. The grid was restarted so every cell shares it.
""")


def s09_headline(prs, charts):
    s = base(prs, 9, "Results  ·  matched site", "Six microphones match twenty-four")
    text(s, 0.55, 1.4, 12.2, 0.55,
         "Every drone frame, before the filter. Within 10° of the true direction. The bar is the seed spread. Pooled seed spread on this site is ±1.2 points.",
         size=15, color=BODY)
    picture(s, charts["headline"], 0.45, 2.0, w=7.5)
    card(s, 8.15, 2.15, 4.6, 4.5)
    text(s, 8.38, 2.35, 4.2, 1.5, [
        [("69.3%  against  70.5%", {"bold": True, "color": INK, "size": 18})],
        "Device 3 at one frame, against all 24. Inside the seed noise.",
    ], size=15, color=BODY, after=8)
    text(s, 8.38, 4.15, 4.2, 2.2, [
        [("73.2%  against  70.1%", {"bold": True, "color": GREEN, "size": 18})],
        "At nine frames, 0.9 s, device 3 moves and all 24 stays put. Depth pays only on the small array.",
        "The searched six, best of 106,183 subsets, scores 52%.",
    ], size=15, color=BODY, after=8)
    notes(s, """
We read the chart top to bottom. Learning beats delay-and-sum by a wide margin. Then 6 matches 24 at one frame. Then nine frames helps the six and not the twenty-four. Then the searched array, which won a geometric contest, is the worst thing we trained.
K=5 is 70.1% and 71.1%, inside the noise. The grid's first reading was a null. Nine frames is where the small array pulls ahead.
How the six was searched: no audio and no training. Every non-coplanar subset of 6 out of 24, which is 106,183 subsets. For each, a noise-free plane wave in 250 directions. The score is the strongest response more than 15° from the true direction, at the 95th percentile. The winner ties all 24 at −3.6 dB on that score and then trains at 52%. The score correlates −0.80 with the arrays it did not select, and −0.36 once its own winner is included. We optimized a proxy and selected the error in the proxy. Device 3 was not that winner. It is one recorder, and it is the six we report.
Three seeds cannot give a permutation p below 0.10. These gaps are point estimates. The history deletion on the next slide is the comparison we tested formally.
""")


def s10_history(prs, charts):
    s = base(prs, 10, "Results  ·  what the frames are doing", "History helps where one frame is weak")
    text(s, 0.55, 1.4, 12.2, 0.7,
         "Same trained weights. The past frames are replaced with a copy of the oldest one we keep, so the only change is how far back the network can see.",
         size=16, color=BODY)
    picture(s, charts["history"], 0.4, 2.15, w=7.6)
    card(s, 8.2, 2.3, 4.55, 4.3)
    text(s, 8.42, 2.5, 4.15, 0.4, "Device 3, nine frames", size=16, color=INK, bold=True)
    bullets(s, 8.42, 3.05, 4.15, 3.3, [
        "Same weights, past deleted: 5.1 points. p = 0.0007 is this comparison. A drop that steady is very unlikely if the past did nothing.",
        "The gain is +0.5 inside 30 m and +21.8 at 90–120 m.",
        "Shuffling or reversing the frames costs about 0.2. The cue is persistence, not order.",
        "The climb ends near 0.8 s.",
    ], size=14, after=8)
    notes(s, """
The mechanism: a drone sits still on the grid while a sidelobe flickers. Persistence does not need the order of the frames, and shuffling or reversing them costs about 0.2. The network agrees.
Why K=5 looked like a null: a wider input costs accuracy before any history is used, and the history earns that cost back. Nine frames is where that term wins, and only on the small array. All 24 at nine frames is 70.1%, the same as one frame. Twenty-four microphones suppress phantoms spatially. Nine frames suppress them in time.
Past 120 m the gain falls back. Two thirds of the labelled frames out there are quieter than typical ambient, so there is nothing to integrate.
What the p-value is. It is not the chance that six beats twenty-four. Those models were trained separately, and three seeds cannot give a permutation p below 0.10.
This p is one paired comparison: the same trained weights, the same frames, once with the real past and once with the past replaced by a copy of the oldest frame we keep. Neighbouring frames are one flight, not thousands of independent trials, so we resampled contiguous blocks. A per-frame test was about 3.5 times too optimistic. Holm adjustment is across the family of claims we tested. p = 0.0007 means that, under that null, a 5.1 point drop this consistent is very unlikely. It is the comparison that survives the correction.
""")


def s11_filter(prs):
    s = base(prs, 11, "After the network", "Inside 10°, we follow at once")
    picture(s, DOCS / "filter_sky.gif", 0.35, 1.42, w=6.4)
    text(s, 0.4, 5.85, 6.3, 1.05,
         "Eight seconds, about 45 m out. Orange is the network on a sidelobe. Green is the filter, inside 10° of GPS.",
         size=14, color=BODY, line=1.05)
    card(s, 6.95, 1.42, 5.85, 5.35)
    text(s, 7.18, 1.58, 5.45, 0.32, "The rule", size=16, color=INK, bold=True)
    text(s, 7.18, 2.0, 5.45, 1.7,
         "A new direction within 10° of the last one is taken at once. A larger jump is taken only after half a second of agreement.",
         size=15, color=BODY, line=1.08)
    text(s, 7.18, 3.85, 5.45, 0.32, "What it changes", size=16, color=INK, bold=True)
    bullets(s, 7.18, 4.28, 5.4, 2.2, [
        "One frame, matched site. 24 microphones go from 70% to 75%. Device 3 goes from 69% to 73%.",
        "Inside 30 m the cells do not move. The gain is farther out.",
        "At nine frames, device 3 stays at 73%. The stack had already learned the smoothness.",
    ], size=14, after=6)
    notes(s, """
The clip is eight seconds of the matched-site flight, device 3, one frame, about 45 m out. White is GPS. Orange is the network hopping to a sidelobe. Green is the filter, and it stays inside the 10° disc. A step inside 10° of the last direction is taken at once. A larger jump waits half a second. Nothing was retrained.
On one frame the matched site goes from 70% to 75% with 24 microphones, and from 69% to 73% with device 3. Inside 30 m nothing moves. At nine frames device 3 stays at 73%.
On the new site, second look, 24 microphones go from 52% to 59% and device 3 from 54% to 58%. Device 3 at nine frames stays at 52%. The farthest bin on device 3 at one frame is 25% ± 4 to 57% ± 33, so that cell is not a number we defend.
Where the raw track is stably wrong, the filter holds the error. It is not a repair for a bad array. A hard hold with no second chance wrecks the score, because some jumps are the model correcting itself.
""")


def s12_paper(prs):
    s = base(prs, 12, "Against the paper", "Their mean error, on the frames they keep")
    text(s, 0.55, 1.42, 12.2, 0.55,
         "Matched site. Their bins. U-Net rows are frames whose last three directions already agree within 10°. Delay-and-sum is every frame. Our U-Net cells are the mean of three seeds.",
         size=15, color=BODY)
    table(s, 0.55, 2.15, [4.6, 2.0, 2.0, 2.3],
          ["Mean angular error", "0–50 m", "50–100 m", "100–200 m"],
          [
              ["Paper, delay-and-sum", "38°", "75°", "33°"],
              ["Ours, delay-and-sum, 24", "40°", "72°", "31°"],
              ["Paper, U-Net", "10°", "13°", "14°"],
              ["Ours, U-Net, 24 mics", "9°", "11°", "46°"],
          ], size=16, rh=0.58)
    card(s, 0.55, 5.15, 12.2, 1.7)
    text(s, 0.8, 5.32, 11.75, 1.35, [
        [("Under 100 m we match them, ", {"bold": True, "color": INK}),
         ("and we are slightly ahead on the frames they keep. Past 100 m their 14° is still ahead of our 46°. The filter is a different cut, and it does not close that gap.", {})],
        [("The new site is a second look at one short flight. ", {"bold": True, "color": INK}),
         ("They lead under 50 m. From 50 m out, device 3 at one frame is 12° and 11° against their 24° and 54°, on 238 frames past 100 m.", {})],
    ], size=15, color=BODY, after=4)
    notes(s, """
We lead with the delay-and-sum check. We reproduce their bars within a few degrees, including the odd dip in the far bin, and their stability rule calls 67.0% of no-drone frames detections against our 68.6%. Geometry, sync offsets, the STFT, the beamformer, the labels, and the metric all agree with an independent implementation.
This table and the hit-rate table do not convert. A mean of 31° in the far bin can still be 2% of frames inside 10°. That is why both exist.
The new-site rows were computed after test 1 had already been scored, on dumps that already existed. It is a second look, not a second sealed trial. 1,592 drone frames, about 2.7 minutes.
""")


def s13_detection(prs):
    s = base(prs, 13, "Detection", "It learns the recording, not the drone")
    card(s, 0.55, 1.5, 4.3, 5.2, fill=DARK, line=None)
    text(s, 0.8, 1.75, 3.85, 0.3, "ONE-MIC DETECTOR", size=13, color="60A5FA", bold=True)
    text(s, 0.8, 2.2, 3.85, 1.1, "0.49", size=72, color="F5F5F4", bold=True)
    text(s, 0.8, 3.55, 3.85, 1.35, "Cross-scene AUROC. Chance is 0.50. One scene held out at a time. ± 0.06.",
         size=16, color="D6D3D1", line=1.12)
    text(s, 0.8, 5.05, 3.85, 1.4,
         "At 95% recall, precision is about 0.79. That is just how often a drone is in the frame.",
         size=15, color="A8A29E", line=1.1)
    points = [
        ("Three recordings", "A scene is one visit's background noise. We have three: 25 November, and two flights on 18 March. More minutes of one field are the same example."),
        ("The small model ties the big one", "Inside one recording, an 86k-parameter detector scores 0.66, level with the U-Net's presence head. One microphone is 0.68. Averaging all 24 does not help."),
        ("A new recording breaks it", "Hold out one of the three and the score is 0.49. On the sealed site, averaging 24 microphones gives 0.64, and one microphone is 0.53. That quiet audio is one minute."),
    ]
    for i, (title, body) in enumerate(points):
        y = 1.5 + i * 1.75
        text(s, 5.15, y, 7.6, 0.35, title, size=18, color=INK, bold=True)
        text(s, 5.15, y + 0.42, 7.6, 1.15, body, size=15, color=BODY, line=1.08)
    notes(s, """
The two detectors, the pre-takeoff audio, the mixed-in recordings, and the 18 test soundscapes are named on the method slide. Here is what they scored.
The 0.49 is the 86k one-microphone network, held out one whole scene at a time. It is chance. The 0.66 beside it is that same network inside one scene, tied with the U-Net's presence head. The 0.64 is that same detector on the sealed site, and only after averaging all 24 microphones. One microphone there is 0.53. The quiet audio is one minute of one scene, so 0.64 is the within-scene result again. We did not find a winner.
This is the negative result, and it is a data fact. The architecture is not the excuse.
Why we did not replace 0.49 with a precision-recall curve. Drone frames are about 78% of the matched site, so they are the majority, not the rare class. A detector that says "drone" on almost every frame has precision near 0.8, and a PR curve would read that prior as skill. At the threshold that keeps 95% of drones, the map models fire on 84–96% of empty frames and precision is 0.78–0.80. AUROC asks the question that does not hide this: does a random drone frame outscore a random quiet frame? Across scenes, no.
The 0.66 from before was two halves of one afternoon.
On 18 other soundscapes the one-microphone detector fires about 22,000 times an hour at 95% recall. Training on those scenes leaves the held-out rate near 10,000 an hour.
Past about 90 m, two thirds of labelled drone frames are quieter than typical ambient in the beamformed map. The localizer has the same wall.
The presence probabilities are not probabilities. The head saturates. A 0.5 threshold means nothing.
""")


def s14_close(prs):
    s = base(prs, 14, "What we would defend", "Three results, and where they stop")
    held = [
        ("Six match twenty-four", "One recorder, no shared clock, is level with the full array. On the matched site that is 69.3% against 70.5% at one frame. On the sealed site, device 3 is again not behind."),
        ("Depth pays where mics are scarce", "On the matched site, nine frames take device 3 to 73% and leave all 24 at 70%. On the sealed site they stay at 52%."),
        ("The filter removes lobe hops", "On one frame, 24 microphones go from 70% to 75% and device 3 from 69% to 73%, past 30 m. Once nine frames are stacked it adds nothing."),
    ]
    stops = [
        ("Past about 90 m", "The labelled drone is often quieter than the background. Their stable-frame 14° is still ahead of us there."),
        ("Inside 30 m", "Worse than the next bin, on both sites. A source metres from a 2 m array is not a plane wave. Untested."),
        ("Detection", "Three scenes cannot teach what a drone is not. The fix is another afternoon in another place."),
    ]
    text(s, 0.55, 1.42, 6.0, 0.3, "HOLDS", size=13, color=GREEN, bold=True)
    text(s, 6.85, 1.42, 6.0, 0.3, "STOPS", size=13, color=AMBER, bold=True)
    for i, (title, body) in enumerate(held):
        y = 1.8 + i * 1.55
        text(s, 0.55, y, 6.0, 0.32, title, size=16, color=INK, bold=True)
        text(s, 0.55, y + 0.34, 6.0, 1.05, body, size=14, color=BODY, line=1.05)
    for i, (title, body) in enumerate(stops):
        y = 1.8 + i * 1.55
        text(s, 6.85, y, 5.9, 0.32, title, size=16, color=INK, bold=True)
        text(s, 6.85, y + 0.34, 5.9, 1.05, body, size=14, color=BODY, line=1.05)
    notes(s, """
We close on the trade a deployment actually faces: 0.9 seconds of latency, or 18 more microphones. They do not add. That trade is the matched site. On the sealed site, nine frames stay at 52%, level with one frame.
What we would run next: ambient from more places, and a steering model that does not assume a plane wave.
One drone, one array, two sites. We say the limit ourselves.
The division of work stays in the report. If we are asked in the room, one sentence is enough.
The filter takes a step inside 10° at once. A larger jump waits half a second.
""")


def overflow(prs) -> int:
    sw, sh = prs.slide_width, prs.slide_height
    n = 0
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            right = shape.left + shape.width
            bottom = shape.top + shape.height
            if right > sw + 20000 or bottom > sh + 20000:
                n += 1
                print(f"overflow slide {i}: right {right / 914400:.2f}in bottom {bottom / 914400:.2f}in")
    return n


def main() -> int:
    charts = save_charts()
    save_unet(DOCS / "unet_structure.png")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(SW * 914400)), Emu(int(SH * 914400))
    s01_title(prs)
    s02_problem(prs)
    s03_data(prs)
    s04_model(prs)
    s05_network(prs)
    s06_hardware(prs)
    s06_map(prs)
    s08_training(prs, charts)
    s09_headline(prs, charts)
    s10_history(prs, charts)
    s11_filter(prs)
    s12_paper(prs)
    s13_detection(prs)
    s14_close(prs)
    if len(prs.slides) != TOTAL:
        raise SystemExit(f"{len(prs.slides)} slides, expected {TOTAL}")
    n = overflow(prs)
    if n:
        raise SystemExit(f"{n} shapes past the slide edge")
    prs.save(OUT)
    charts["training"].unlink()
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
