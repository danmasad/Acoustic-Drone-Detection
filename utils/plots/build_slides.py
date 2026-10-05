"""Build the project presentation (docs/presentation.pptx) from docs/report.md.

Numbers are copied from that report (5 October 2026). Two readings stay on
separate slides: the paper's mean angular error, and our share inside 10°.

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/build_slides.py
"""
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
OUT = ROOT / "docs" / "presentation.pptx"

FONT = "Helvetica Neue"
BG, DARK = "FAFAF9", "0A0A0A"
INK, BODY, MUTED, LINE, CARD = "1C1917", "44403C", "78716C", "E7E5E4", "FFFFFF"
BLUE, GREEN, RED, AMBER, GREY = "2563EB", "16A34A", "DC2626", "D97706", "D6D3D1"
SW, SH = 13.333, 7.5
TOTAL = 8


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
    return box(slide, x, y, w, h, fill, line, MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)


def text(slide, x, y, w, h, paras, size=16, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, after=6, line=1.12):
    """paras: str, or list of paragraphs; a paragraph is a str or a list of (text, style) runs."""
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


def bullets(slide, x, y, w, h, items, size=16, dot=BLUE, after=10):
    paras = []
    for it in items:
        runs = [("●   ", {"color": dot, "size": size - 6})]
        runs += [(it, {})] if isinstance(it, str) else it
        paras.append(runs)
    return text(slide, x, y, w, h, paras, size=size, after=after)


def hline(slide, x1, y1, x2, y2, color=LINE, width=0.75, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                   Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    return c


def base(prs, n, kicker, title, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK if dark else BG)
    if kicker:
        text(s, 0.7, 0.45, 10, 0.3, kicker.upper(), size=11, color=BLUE, bold=True)
    if title:
        text(s, 0.7, 0.72, 11.9, 0.8, title, size=30, color=INK, bold=True)
    if n:
        text(s, 0.7, 7.02, 8, 0.25, "Hearing drones in 3D  ·  Livshits & Masad", size=9, color=MUTED)
        text(s, 11.63, 7.02, 1.0, 0.25, f"{n} / {TOTAL}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def picture(slide, path, x, y, w=None, h=None, crop=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    if crop:
        l, t, r, b = crop
        pic.crop_left, pic.crop_top, pic.crop_right, pic.crop_bottom = l, t, r, b
        pw, ph = Image.open(path).size
        vis_w, vis_h = pw * (1 - l - r), ph * (1 - t - b)
        if w is not None:
            pic.width = Inches(w)
            pic.height = Inches(w * vis_h / vis_w)
        else:
            pic.height = Inches(h)
            pic.width = Inches(h * vis_w / vis_h)
    return pic


def stat(slide, x, y, w, big, label, color=INK, big_size=34, h=1.45):
    card(slide, x, y, w, h)
    text(slide, x + 0.25, y + 0.18, w - 0.5, 0.65, big, size=big_size, color=color, bold=True)
    text(slide, x + 0.25, y + 0.18 + big_size / 72 * 1.25, w - 0.5, h - 0.9, label,
         size=12, color=BODY, after=2)


# ---------------------------------------------------------------------------------------------
# Hit-rate cells are colored from the leading number. Degree tables must pass
# color_cells=False: a small angle is the better score, and this scale would invert it.

def tone(cell):
    head = cell.split("±")[0].strip()
    if not head or not head[0].isdigit():
        return MUTED
    v = int(head)
    if v >= 65:
        return GREEN
    if v < 40:
        return RED
    return INK


def table(slide, x, y, widths, headers, rows, size=13, rh=0.46, color_cells=True, pct=False):
    """rows: list of lists of strings. First column is the row name.

    `pct` appends % to numeric score cells. The closing frame-count row is left as a count.
    """
    cx = x
    for j, h in enumerate(headers):
        text(slide, cx, y, widths[j] - 0.06, 0.46, h.split("\n"), size=11, color=MUTED,
             bold=True, align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, after=0, line=1.05)
        cx += widths[j]
    hline(slide, x, y + 0.50, x + sum(widths), y + 0.50, color=MUTED)
    for i, row in enumerate(rows):
        yy = y + 0.60 + i * rh
        cx = x
        last = i == len(rows) - 1 and row[0].startswith("drone frames")
        for j, cell in enumerate(row):
            shown = cell + "%" if pct and not last and j > 0 and cell[:1].isdigit() else cell
            color = MUTED if last else (INK if j == 0 else tone(cell) if color_cells else INK)
            text(slide, cx, yy, widths[j] - 0.06, 0.34, shown,
                 size=size if not last else 11, color=color, bold=(j == 0 and not last),
                 align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
            cx += widths[j]
        if i < len(rows) - 1:
            hline(slide, x, yy + rh - 0.08, x + sum(widths), yy + rh - 0.08, width=0.5)
    return y + 0.60 + len(rows) * rh


def s01_title(prs):
    s = base(prs, None, None, None, dark=True)
    text(s, 0.75, 1.45, 6.3, 0.28, "APPLIED DEEP LEARNING  ·  AUDIO", size=12,
         color="60A5FA", bold=True)
    text(s, 0.75, 1.85, 6.1, 1.7, ["Hearing drones", "in 3D"], size=46, color="F5F5F4",
         bold=True, after=0, line=0.98)
    text(s, 0.75, 4.05, 5.9, 1.35,
         "Six microphones on one recorder match the full array. Extra frames help only "
         "where microphones are scarce.",
         size=18, color="A8A29E", line=1.15)
    text(s, 0.75, 5.7, 5.9, 0.32, "Shai Livshits  ·  Dan Masad", size=16, color="F5F5F4")
    text(s, 0.75, 6.1, 5.9, 0.28, "5 October 2026", size=13, color="78716C")
    picture(s, FIG / "flight_hemisphere_mar18_f2.gif", 7.05, 1.55, w=5.55)


def s02_problem(prs):
    s = base(prs, 2, "01   ·   The problem", "A direction, ten times a second")
    bullets(s, 0.7, 1.58, 7.15, 4.05, [
        "Drones are small, and they often fly with no radio link. That defeats a camera in poor visibility, and a radio detector outright.",
        "Sound is the cheap fallback. Rotors are loud, and the sound is harmonically distinctive.",
        "From a ground array we want a report, ten times a second: is a drone audible, and in which direction.",
        "Published systems use 24 or more synchronized microphones. Six in one off-the-shelf recorder would cost about four times less.",
    ], size=16, after=12)
    card(s, 8.15, 1.58, 4.5, 4.05, fill=DARK, line=None)
    text(s, 8.4, 1.78, 4.05, 0.26, "THE QUESTION", size=12, color="60A5FA", bold=True)
    text(s, 8.4, 2.12, 4.05, 1.05,
         "Rodríguez et al. (2026) leave temporal context as untested future work. Can a few frames of history let six microphones match twenty-four?",
         size=14, color="D6D3D1", line=1.12)
    text(s, 8.4, 3.28, 4.05, 0.26, "WHAT WE FOUND", size=12, color="60A5FA", bold=True)
    text(s, 8.4, 3.60, 4.05, 1.8,
         "They can, and the history is not why. Six on one recorder match the full array on every drone frame. Extra frames help only where microphones are scarce.",
         size=14, color="F5F5F4", line=1.12)
    text(s, 0.7, 5.85, 12.0, 0.95,
         "Their dataset and their splits. One DJI Air 3, four recorders of six, 48 kHz, GPS at 10 Hz. "
         "Training is 52 minutes at site 1. Test 2 is 20 minutes at the same site. Test 1 is 3 minutes "
         "at another site and stayed sealed through development. Targets are 10° discs.",
         size=14, color=BODY)


def s03_array(prs):
    s = base(prs, 3, "02   ·   The array", "Unambiguous only below 1.47 kHz")
    stats = [
        ("1.47 kHz", "Unambiguous only below this."),
        ("2.2 kHz", "The maps run to here, past that band."),
        ("3.6 dB", "The phantom all 24 microphones still leave."),
    ]
    w, gap = 3.85, 0.18
    for i, (big, label) in enumerate(stats):
        stat(s, 0.7 + i * (w + gap), 1.48, w, big, label, big_size=22, h=1.15)
    text(s, 0.7, 2.85, 6.9, 3.9,
         "Sixteen sub-bands on a 23 by 91 grid at 4°. A sidelobe can line up almost as well as the source. "
         "All 24 has a coplanar recorder and a pair 0.17 m apart, so eighteen further microphones buy nothing.\n\n"
         "Every panel is the same map. Highlighted channels are the array that was trained. "
         "The new site is those microphones after they were moved. "
         "Lobe hops are what the filter removes. The smooth miss inside 30 m is the near field, and the filter does not fix it.",
         size=14, color=BODY, line=1.15)
    picture(s, FIG / "array_tested.png", 8.05, 2.72, w=4.5)


def s04_model(prs):
    s = base(prs, 4, "03   ·   The model", "A U-Net on the map, with two measured changes")
    cards = [
        ("Architecture",
         "A depth-3 U-Net, 3×3 convolutions, circular padding on azimuth. A per-pixel mask, and a presence head pooled from the middle. It reads the 23 by 91 map."),
        ("Causal history",
         "Temporal context is K frames stacked on the channel axis, newest last, so the stack is causal. The runs use K = 1, 5, and 9."),
        ("Training",
         "Adam at 5×10⁻³, class-weighted cross-entropy, early stopping on the test-2 hit rate, patience 10. 32 base filters rather than 64: faster, and 72% against 63% of frames within 10°. A leaky presence head, after a plain ReLU died in one seed."),
        ("The baseline",
         "Brightest-pixel delay-and-sum, on the same frames. A separate 1-D CNN, 86k parameters, detects from one microphone, so it can be tested on audio we did not render as a map."),
    ]
    for i, (title, body) in enumerate(cards):
        x = 0.7 + (i % 2) * 6.2
        y = 1.58 + (i // 2) * 2.62
        card(s, x, y, 6.0, 2.48)
        text(s, x + 0.28, y + 0.2, 5.45, 0.36, title, size=16, color=INK, bold=True)
        text(s, x + 0.28, y + 0.68, 5.45, 1.6, body, size=15, color=BODY, line=1.12)


def s05_filter(prs):
    s = base(prs, 5, "04   ·   The kinematic filter", "After the mask, not inside the network")
    stats = [
        ("21 m/s", "A DJI Air 3 in sport mode, the published speed."),
        ("2.1 m", "How far that allows in one 0.1 s frame."),
        ("+ 4°", "One grid cell, on top of the sky-angle cap."),
        ("0.5 s", "Five frames outside the cap before a jump is kept."),
    ]
    w, gap = 2.9, 0.16
    for i, (big, label) in enumerate(stats):
        stat(s, 0.7 + i * (w + gap), 1.58, w, big, label, color=BLUE, big_size=28, h=1.48)
    card(s, 0.7, 3.22, 6.05, 3.55)
    text(s, 0.95, 3.36, 5.55, 0.32, "When the stored direction moves", size=15, color=INK, bold=True)
    bullets(s, 0.95, 3.74, 5.55, 2.85, [
        "It runs after the mask, on the decoded direction. It is not an input channel, and not a reason to retrain.",
        "Inside the cap the stored direction updates at once. The cap is the sky angle of 2.1 m at the GPS range, plus one 4° cell.",
        "A hard hold with no second chance wrecks the score. Some large jumps are the model correcting itself.",
        "A fixed 16° cap, with no range, landed within about a point of this rule on the same-site 24-microphone run.",
    ], size=13, after=8)
    card(s, 6.95, 3.22, 5.7, 3.55)
    text(s, 7.2, 3.36, 5.25, 0.32, "What the history was doing", size=15, color=INK, bold=True)
    text(s, 7.2, 3.70, 5.25, 0.36, "Hit rate, on weights already trained. Not the paper’s mean angle.",
         size=13, color=MUTED)
    bullets(s, 7.2, 4.12, 5.25, 2.45, [
        "Corrupting the past of the K = 5 models costs 4.3 points. The climb ends near 0.8 s.",
        "Deleting history from the K = 9 weights costs 5.1 points, block-resampled, Holm-adjusted p = 0.0007.",
        "About +0.5 inside 30 m, +8.7 at 60–90 m, and +21.8 at 90–120 m. Order does not matter. The cue is persistence.",
    ], size=13, after=8)


def s06_paper(prs):
    s = base(prs, 6, "05   ·   Against the paper", "Mean angular error, on their terms")
    text(s, 0.7, 1.32, 12.0, 0.48,
         "Their metric only. U-Net rows are stable frames: the last three directions within 10°. "
         "Delay-and-sum is every frame. The lower block is the new site, a second look. "
         "Matched-site U-Net cells are the range over three seeds.",
         size=13, color=BODY, line=1.05)
    widths = [4.35, 2.15, 1.8, 1.8, 1.8]
    headers = ["mean angular error", "site", "0–50 m", "50–100 m", "100–200 m"]
    rows = [
        ["Paper, delay-and-sum", "matched", "38°", "75°", "33°"],
        ["Ours, delay-and-sum, 24", "matched", "40°", "72°", "31°"],
        ["Paper, U-Net, stable", "matched", "10°", "13°", "14°"],
        ["Ours, U-Net 24, stable", "matched", "7–10°", "9–12°", "31–74°"],
        ["Paper, delay-and-sum", "new, second look", "8°", "29°", "92°"],
        ["Ours, delay-and-sum, 24", "new, second look", "9°", "23°", "83°"],
        ["Paper, U-Net, stable", "new, second look", "12°", "24°", "54°"],
        ["Ours, 24 mics, K = 1, stable", "new, second look", "16±4°", "10±3°", "20±9°"],
        ["Ours, device 3, K = 1, stable", "new, second look", "14°", "12±3°", "11±2°"],
    ]
    y0, rh = 1.84, 0.42
    table(s, 0.7, y0, widths, headers, rows, size=12, rh=rh, color_cells=False)
    hline(s, 0.7, y0 + 0.60 + 4 * rh - 0.08, 0.7 + sum(widths), y0 + 0.60 + 4 * rh - 0.08,
          color=BLUE, width=1.5)
    text(s, 0.7, 6.20, 12.0, 0.78, [
        "On the matched site, under 100 m, we are level and slightly ahead on the frames they keep: "
        "7–10° and 9–12° against 10° and 13°. Past 100 m their 14° still beats our stable-frame mean, "
        "31–74°. The filter is not this cut. On the new site, a second look, they lead under 50 m and we "
        "lead from 50 m out — device 3 is 12±3° and 11±2° against their 24° and 54°, on 1,592 frames, "
        "238 past 100 m.",
        [("Delay-and-sum reproduces their bars within a few degrees. Their rule calls 67.0% of no-drone "
          "frames detections; ours calls 68.6%.", {"color": MUTED, "size": 12})],
    ], size=13, color=INK, after=1, line=1.02)


def s07_variants(prs):
    s = base(prs, 7, "06   ·   Against ourselves", "Every drone frame, raw then filtered")
    text(s, 0.7, 1.48, 12.0, 0.42,
         "Matched site. Not their bins and not their mean. Share of drone frames inside 10°, our 30 m bins. "
         "Each cell is raw, then filtered. ± is the seed spread, omitted under 2. Pooled spread here is ±1.2 points.",
         size=13, color=BODY, line=1.02)
    widths = [2.55, 1.6, 1.6, 1.55, 1.7, 1.75, 1.2]
    headers = ["inside 10°\nraw → filtered", "0–30 m", "30–60 m", "60–90 m", "90–120 m", "120–180 m", "all"]
    rows = [
        ["Delay-and-sum, 24", "50→52", "33→36", "18→21", "3→0", "2→0", "33→35"],
        ["24 mics, K = 1", "66±5→66±5", "79→83±3", "73→82±6", "57±3→74±12", "27±11→55±13", "70±2→76"],
        ["24 mics, K = 5", "63→62", "79±3→81±2", "75→77±3", "60±2→64±6", "27±6→34±12", "70→72"],
        ["Device 3, K = 1", "69±3→70±2", "75→81±2", "68→77", "56→64±20", "25±4→65±11", "69→75±2"],
        ["Device 3, K = 9", "72±2→72±2", "78±3→78±4", "73±3→74±4", "71±12→74±12", "24±6→22±4", "73→73±2"],
    ]
    table(s, 0.7, 1.92, widths, headers, rows, size=12, rh=0.42, color_cells=False)
    card(s, 0.7, 4.72, 11.95, 0.70, line=AMBER)
    text(s, 0.92, 4.80, 11.55, 0.54,
         "Not a comparison with the paper. Delay-and-sum, matched site, our reading of the sideways miss, "
         "range × sin(angle): about 3 m under 30 m, and about 64 m in the far bin. That far-bin entry is "
         "the bin’s 32° mean. A 10° error is about 5 m at 30 m and 17 m at 100 m.",
         size=13, color=INK, line=1.05)
    text(s, 0.7, 5.52, 12.0, 1.42, [
        [("Six microphones match twenty-four. ", {"bold": True, "color": INK}),
         ("At one frame, 69.3% ± 0.7 against 70.5% ± 1.5, inside seed noise, and at five frames still inside it. "
          "At nine frames, 0.9 s, device 3 reaches 73.2% ± 1.4 and all 24 stays at 70.1% ± 2.7. Depth pays only "
          "on the small array. The searched six, out of 106,183, score 52.1% ± 0.8.", {})],
        [("The filter, on this reading. ", {"bold": True, "color": INK}),
         ("One frame takes 24 microphones from 70% to 76% and device 3 from 69% to 75%. The gain is in the far bins; "
          "inside 30 m the cells do not move. Once five or nine frames are stacked it adds about a point, and device 3 "
          "at K = 9 stays at 73%. New site, a second look: 52% to 61% and 54% to 61%; device 3 at K = 9 stays at 52%. "
          "Where the raw track is stably wrong, the filter holds the error.", {})],
    ], size=13, after=4, line=1.05)


def s08_limits(prs):
    s = base(prs, 8, "07   ·   Limits", "What these numbers cannot support")
    top = [
        ("Past about 90 m",
         "The labelled drone is often quieter than ambient in the beamformed map, and two thirds of labelled positives there are quieter than typical ambient. Early stopping on a pooled hit rate never selected for that bin. On the matched site, their stable-frame 14° remains ahead of us there."),
        ("Inside 30 m",
         "The learned cells sit below the next bin, and the filter leaves them there. On the new site, a second look, the closest 30 m stay near 30% inside 10°. A source metres from a 2 m array is not a plane wave. Untested."),
        ("Set aside",
         "Polar discs, Tversky loss, and the rest of the array list were measured. None changes the claim above. Sub-band count moves the hit rate by about half a point."),
    ]
    for i, (title, body) in enumerate(top):
        x = 0.7 + i * 4.15
        card(s, x, 1.58, 3.98, 2.42)
        text(s, x + 0.2, 1.72, 3.58, 0.5, title, size=15, color=INK, bold=True)
        text(s, x + 0.2, 2.26, 3.58, 1.58, body, size=13, color=BODY, line=1.05)
    bottom = [
        ("Detection fails across scenes",
         "Held out one scene at a time, the per-channel detector scores 0.488 ± 0.056 AUROC. The 0.662 was two halves of one afternoon. On the sealed new site the presence score reads 0.639 AUROC. Training negatives grew from 4.8 to 8.2 minutes and from one scene to three. That is still not enough to learn a detector."),
        ("One drone, one dataset, two sites",
         "The sealed new-site score, once, already said device 3 is not worse than all 24: 54.4% ± 4.6 against 52.2% ± 5.5 at one frame. The seed spread widens to ±5–8 points, and delay-and-sum wins the pool, because most of that flight is inside 90 m. The paper bins and the filter on that flight are a second look at those dumps."),
    ]
    for i, (title, body) in enumerate(bottom):
        x = 0.7 + i * 6.3
        card(s, x, 4.16, 6.1, 2.62)
        text(s, x + 0.24, 4.30, 5.62, 0.36, title, size=15, color=INK, bold=True)
        text(s, x + 0.24, 4.74, 5.62, 1.85, body, size=14, color=BODY, line=1.05)


def main() -> int:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(SW * 914400)), Emu(int(SH * 914400))
    for f in (s01_title, s02_problem, s03_array, s04_model, s05_filter,
              s06_paper, s07_variants, s08_limits):
        f(prs)
    assert len(prs.slides) == TOTAL
    prs.save(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
