---
title: "Hearing Drones in 3D: What a Cheap Microphone Array Can and Cannot Do"
author: "Shai Livshits, Dan Masad"
date: ""
geometry: margin=0.7in
fontsize: 10pt
colorlinks: true
header-includes: |
  \usepackage{titlesec}
  \usepackage{titling}
  \usepackage{enumitem}
  \usepackage{caption}
  \usepackage{etoolbox}
  \setlength{\droptitle}{-5em}
  \posttitle{\par\end{center}}
  \titlespacing*{\section}{0pt}{2pt}{0pt}
  \titlespacing*{\subsection}{0pt}{2pt}{0pt}
  \setlength{\parskip}{0.5pt}
  \linespread{0.98}
  \captionsetup{font=footnotesize,skip=1pt}
  \AtBeginEnvironment{longtable}{\footnotesize}
  \setlength{\LTpre}{2pt}
  \setlength{\LTpost}{2pt}
  \setlength{\textfloatsep}{4pt}
  \setlength{\intextsep}{4pt}
  \setlist[itemize]{itemsep=0pt,topsep=1pt,parsep=0pt,leftmargin=1.1em}
---

**Category:** Applied — Audio / Convolutional Networks / Defense & Security.
**Code:** [github.com/danmasad/Acoustic-Drone-Detection](https://github.com/danmasad/Acoustic-Drone-Detection)

## 1. Motivation and problem definition

Drones are small, quiet targets that often fly without a radio link, which defeats cameras in poor
visibility and radio detectors outright. Sound is a cheap fallback: rotors are loud and harmonically
distinctive. From a ground array we want a system reporting, ten times per second, whether a drone is
audible and in which direction — azimuth (compass bearing) and elevation (angle above the horizon).
The obstacle is hardware: published systems use 24 or more synchronized microphones, and if six in
one off-the-shelf recorder did as well, a deployable sensor would cost about four times less. Our
reference paper (Rodríguez et al., 2026) names temporal context as its own untested future work, so
our question was:

> **Can temporal context let a cheap 6-microphone array match a 24-microphone one?**

The premise turned out to be wrong in an interesting way, which is what this report is about.

## 2. Related work

Classical localization steers an array and reads off the strongest direction. The canonical method,
and our baseline, is **delay-and-sum (DAS) beamforming**: guess a direction, undo the per-microphone
delays the geometry implies for it, sum, and measure energy — a correct guess adds the copies in
phase and spikes. Sweeping all candidates turns one frame into an **energy map** over azimuth and
elevation whose brightest pixel is the estimate. Refinements (GCC-PHAT, SRP-PHAT, MVDR, MUSIC)
reweight the sum; all degrade at low signal-to-noise ratio and all share one failure — a
**sidelobe**, a direction whose delays align almost as well as the true one, produces a phantom peak
a single frame cannot distinguish from the source, and thinning an array worsens it.

Learned approaches usually regress direction from multichannel audio. Our reference paper instead
treats the map as an *image* and localization as semantic segmentation: a U-Net outputs a mask over
the azimuth–elevation grid, and its centroid is the estimate. We adopt this because it makes the
array a *preprocessing* choice rather than an architectural one — the network sees a map, not
channels, so one architecture trains on any subset, which is what our experiment needs. The paper
uses a single frame, evaluates detection only as a stability proxy on five minutes of no-drone audio,
and released its dataset and beamforming scripts but no training code, so everything here is
reimplemented from its description.

## 3. Data

The Western Sydney University UAV Acoustic Localization Dataset (CC BY-SA 4.0, 24 GB): open-field
recordings of a DJI Air 3 from a 24-microphone array — four independent recorders of six microphones
each, 48 kHz — with GPS flight logs at 10 Hz and per-recorder sample offsets. Splits follow the
paper: 52 minutes of training across two sessions at site 1, 20 minutes of test 2 at the same site,
3 minutes of test 1 at a different site. **Test 1 was sealed** through all development and model
selection. Ground truth comes from the logs, where row *k* of the 10 Hz log is frame *k* of the
segment — that identity is the whole alignment scheme — giving each 100 ms frame an azimuth,
elevation and range. Segmentation targets are 10°-radius discs; presence is a binary label.

Negative (no-drone) audio is the weak point: the release labels about 5 minutes. We mined the
pre-takeoff portion of every recording, which the dataset's own splits never use, verified by
beamformed peak contrast that those windows are ambient, and dropped a 20 s margin for motor
spin-up. Training negatives went from 4.8 to 8.2 minutes and from one scene to three — and section 5.5
shows that this is still not enough, and why.

## 4. Method

**Features.** One pass over the audio caches every frame's per-microphone spectra (381 bins over
200–4000 Hz, 3.4 GiB), so every downstream choice — microphone subset, direction grid, sub-band
count, frame stacking — is a re-steer over the cache rather than another read of 24 GB.
Frequency-domain DAS turns cached spectra into maps of 23 elevations × 91 azimuths at 4°, averaged
into 16 sub-bands over 200–2200 Hz. Keeping bands separate matters because a drone is a harmonic
stack: its true direction lights up in every band, a sidelobe artefact does not.

**Model.** A depth-3 U-Net (PyTorch), 3×3 convolutions, two heads: a per-pixel mask and a pooled
presence logit. Two departures from the reference: circular padding on azimuth, since the map's left
and right edges are the same bearing, and the presence head itself, which is what makes
precision–recall measurable. Temporal context is K frames stacked on the channel axis, current frame
last, so the stack is causal and the label belongs to the newest frame.

**Hyperparameters.** Taken from the paper rather than re-searched, since re-searching per condition
would confound the comparison: depth 3, Adam at lr 5e-3, class-weighted cross-entropy. Two
deviations, both measured: 32 base filters instead of 64 (2.3× faster *and* better, 72% against 63%
within 10°) and a `LeakyReLU` presence head after the plain ReLU died in one seed. Early stopping on
the test-2 hit rate, patience 10, best checkpoint restored, identically in every condition — which is
what makes the cells comparable. The **baseline** is classical DAS with brightest-pixel decoding, on
the same frames with the same metrics.

**A separate detector.** The presence head reads a beamformed map, so injected external audio would
carry a spatial signature our own rendering invented. We therefore also built a per-channel detector:
a 1-D CNN (86k parameters) over one microphone's spectrum, frame decisions by averaging per-channel
logits. Exploiting no cross-channel structure, it can eat mono audio unrendered.

## 5. Experiments and results

All numbers are on test 2 unless stated. Direction is undefined without a drone, so localization is
scored on drone frames only, by **angular error** (the great-circle angle between predicted and true
direction, folding both axes into one; median, the distribution being heavy-tailed) and **within
10°** (frames inside the dataset's own 10° label tolerance). **K** is the number of stacked frames.
Headline cells use three seeds with nothing else changed; "±" is the seed standard deviation, "pp" is
percentage points. The pooled within-cell seed spread is **±1.2 pp** and is the yardstick throughout
— a smaller gap is not a result. Best per column in bold.

### 5.1 Does our pipeline reproduce the reference paper?

Their results are published as figures, so comparison meant adopting their conventions: their
distance bins, the mean rather than median error (our median against their mean would flatter us by
~16°), and their detection rule, which thresholds no energy but calls a frame detected when the last
three direction estimates agree within 10°. Mean angular error for DAS on all 24 microphones over
their bins of 0–50, 50–100 and 100–200 m is **38°, 75°, 33° for them against 40°, 72°, 31° for us**,
and their stability rule wrongly calls 67.0% of no-drone frames detections against our 68.6%. We
reproduce their baseline within a few degrees in every bin, including its non-monotonicity where
100–200 m beats 50–100 m — our strongest end-to-end check, exercising array geometry, per-recorder
offsets, the STFT, the beamformer, label alignment and the metrics at once against an independent
implementation. Our U-Net matches their figures under 100 m under their own stability rule (7–10° and
9–12° against their 10° and 13°); beyond 100 m it does not, and section 5.6 shows that bin has no acoustic
evidence in it.

### 5.2 Headline: microphone count and temporal context

Rows are microphone subsets: all 24; `device 3`, one off-the-shelf recorder's six microphones,
needing no cross-device synchronization; and `best-6`, the six our geometric search called optimal
(see below). Only the array and K change.

*Drone frames within 10°, mean ± seed sd over three seeds. Last column: what context buys.*

| Array | Mics | Recorders | K=1 | K=5 | K=5 − K=1 |
|---|---|---|---|---|---|
| all 24 | 24 | 4 | **70.5% ± 1.5** | 70.1% ± 0.9 | −0.4 |
| device 3 | 6 | 1 | 69.3% ± 0.7 | **71.1% ± 1.8** | **+1.8** |
| best-6 (searched) | 6 | 2 | 52.1% ± 0.8 | 52.8% ± 1.2 | +0.6 |

**Six microphones on one recorder match twenty-four**: the gap is −1.2 pp at K=1 and +1.0 at K=5,
both inside seed noise. Temporal context appears not to help, and that reading is wrong, for the
reason section 5.3 gives. Prompted by it we trained a fourth cell, `device 3` at **K=9** — nine frames, 0.9 s
— which reaches **73.2% ± 1.4**: +3.9 pp over the same array at K=1 and +2.1 over K=5, the first gain
here that clears the ±1.2 pp yardstick. Nine frames of a cheap
recorder also beat 24 synchronized microphones fed one frame at a time (70.5% ± 1.5) — not a
statement that six microphones beat twenty-four, since we never trained all 24 at K=9, but a
statement that *0.9 s of latency substitutes for 18 microphones*, which is the trade a deployment
actually faces.

**The third row is a warning about proxies.** `best-6` was chosen before any training, by scoring
all 106,183 non-coplanar 6-microphone subsets on a purely geometric criterion — the strongest phantom
peak more than 15° off the true direction, in dB, over 250 simulated directions — and it came *last*,
18 pp behind one off-the-shelf recorder. The criterion is not useless, it was *over-optimized*:
across every array it did not select it predicts accuracy at **−0.80**, and including its own winner
collapses that to −0.36. It assumes free-field propagation, exact geometry and perfect
synchronization, so maximizing `true quality + model error` over 106,183 candidates selects hard on
the error term, and the winner of a large search is where the proxy deserves least trust.
Cross-recorder synchronization was our first suspect and is dead: two of the three best arrays we
tried span two recorders. Design notes 22 has all thirteen.

### 5.3 What temporal context actually buys

A model handed five frames can ignore four of them, so we corrupt the history of *already-trained*
K=5 models at test time, with no retraining. `freeze j` keeps the most recent *j* frames real and
replaces earlier ones with a copy of the oldest kept frame, holding weights, parameter count and
input shape fixed while **history depth** varies: j=5 is unmodified, j=1 has no history.

*Within 10°, averaged over all nine K=5 runs (3 arrays × 3 seeds). Increments are paired — same
weights, two depths — so "±" is the standard error over runs.*

| History depth | j=1 | j=2 | j=3 | j=4 | j=5 |
|---|---|---|---|---|---|
| Within 10° | 60.4% | 61.8% | 63.2% | 64.1% | **64.7%** |
| Gain (pp) | — | +1.49 ± .21 | +1.31 ± .22 | +0.97 ± .20 | +0.53 ± .07 |

**History is worth +4.3 pp** and every step is significant, though the increments are halving. That
was the gate for training K=9, and it under-predicted: extrapolating the decelerating increments
promised about +1.0 pp for frames six to nine, and the trained cell delivered +2.1. Repeating the
intervention on those K=9 weights then found where the climb ends — eight significant increments
and a ninth of +0.19 ± 0.11 pp, inside noise. **History saturates at about 0.8 s**, measured rather
than assumed, which is what a deployment needs in order to budget its latency. Two further
controls (figure 1d): shuffling the five frames costs 0.1 pp and reversing them 0.2 pp, so the model reads the
past but is indifferent to its *order* — as the physics demands, since beyond about 70 m the true
direction moves less than one grid cell across a whole stack. The usable cue is persistence, which is
order-invariant: a drone persists while sidelobes flicker.

This resolves the apparent null in section 5.2. A K=5 model with its history deleted scores 67.5% on all 24
where a K=1 model trained on single frames scores 70.5%: five times the input channels costs 3.0 pp
in optimization and overfitting, and history earns back 2.6 pp. Both are true at once — **the
mechanism works, and stacking pays for itself and little more.** Pairing on frame identity, which
removes between-frame variance and is far more sensitive than comparing pooled means, shows *where*
it pays: K=5 − K=1 is negative under 30 m and positive from 60 m out, most strongly on the array with
the least spatial information (+8.4 pp at 60–90 m, +10.6 pp beyond 120 m on `best-6`). Context helps
where one frame is ambiguous and costs more than it returns where one frame suffices.

### 5.5 Detection: the result we had, and why we withdrew it

Detection is scored on every frame by **AUROC**: the probability that a random drone frame outscores
a random no-drone frame, 0.5 being indistinguishable and no threshold needed, which matters because
81% of frames contain a drone.

Our first numbers were 0.586–0.727 for the map model's presence head, with a seed spread of
0.07–0.22 and one run at 0.288 (worse than chance), and a stable 0.662 for the per-channel detector.
Then we looked at which segments they came from. The training and test ambient are `no_drone_1` and
`no_drone_2`: **two halves of one continuous recording** of the same field, the same afternoon, the
same array, so anything learned about that soundscape transfers to its own second half for free. Our
three scenes — Nov 25 flight 11, Mar 18 flight 1, Mar 18 flight 2 — let the fold boundary be a
*scene* boundary instead, each held out in turn, with early stopping monitored on interleaved blocks
of the training scenes and never on the held-out one.

*AUROC on the held-out scene; chance is 0.5. "Confusers" mixes ESC-50 environmental sound into
training frames of both classes, so that "contains an added sound" cannot become the cue.*

| Held-out scene | Clean | + ESC-50 confusers |
|---|---|---|
| A — different session | 0.441 | 0.347 |
| B — same session as C | 0.551 | **0.768** |
| C — same session as B | 0.472 | 0.662 |
| **mean ± sd over folds** | **0.488 ± 0.056** | 0.592 ± 0.219 |

**Cross-scene, detection is at chance** — not weak, absent. The 0.662 was measuring the recording,
not the drone. The one fold from a *different session* is at or below chance in every variant, while
the two folds sharing a session with their training data reach 0.66–0.77: this detector generalizes
across flights, not across sites. ESC-50 confusers buy +0.10 on average, entirely in the folds where
something already worked.

Our own 5.6 minutes of ambient can only resolve 10.7 false alarms per hour, so we also scored the
detector on **DEMAND**: 90 minutes of continuous ambient from 18 environments at our own 48 kHz,
level-matched to ours, at thresholds fixed on our own drone frames. At the threshold catching 95% of
drones it fires on 69% of DEMAND frames — **24,881 false alarms per hour**, against 93% of our own
ambient — and backing off to 50% recall still leaves 6,733, about two per second. No operating point
makes this a detector, and the rate is the same on our audio as on eighteen unrelated soundscapes, so
it is not an artefact of our field or our microphones. Consistently, that same detector fires on 76% of ESC-50
sounds mixed into our ambient, averaged over the three held-out scenes, and on 76% of the eight
classes we named as drone-like in advance (helicopter, airplane, chainsaw, engine, vacuum cleaner,
washing machine, hand saw, wind) — a detector this saturated fires on nearly everything.

### 5.6 How far can it hear, and how long does it take to decide?

A pooled AUROC answers neither. Both come out of the same per-frame scores with no retraining: fix
the threshold on no-drone frames at a false-alarm budget and measure recall per range bin, then
average each score over a causal window and watch false alarms trade against latency.

![(a) Detection against range at 107 false alarms/hour, the finest rate 5.6 minutes of ambient can express; the right axis converts per-frame recall into the wait for a first alarm. (b) Labelled drone frames quieter than a typical no-drone frame, by the baseline's beamformed peak energy. (c) Averaging each model's presence score over a causal window, latency being (window − 1) × 100 ms. (d) Trained models evaluated with 1 to *j* frames of real history, weights and input width fixed: solid for the nine K=5 runs, dashed for the K=9 cell of section 5.3, where the climb stops.](../figures/report_figure_test_2.png){width=93%}

**About 90 m, in under a second.** Per-frame recall of 30% sounds like failure and is the wrong
unit: frames arrive at 10 Hz and an overflight lasts tens of seconds, so 30% per frame is a first
alarm after 0.3 s. Under 90 m every array alarms within about a second; beyond it none alarm at all,
and panel (b) says why — two thirds of the labelled positives out there are quieter than typical
ambient. Labels come from GPS, not audibility, so past 90 m we train and score against frames whose
strongest map peak is not the drone, which bounds what a peak- or mask-decoded model can do here and
explains the section 5.1 gap.

**Time pays for detection too.** A causal average lifts AUROC from 0.727 to 0.773 at 2.9 s of
latency, monotonically, in every seed and array. The mechanism differs from stacking: stacking hands
the network five raw maps and hopes, integration averages its output, and presence is constant across
a window while direction is not — averaging repeated measurements of a fixed quantity, the case where
it must help. Caveat: every segment is homogeneous, so no window straddles a drone's arrival, which
is where a deployed system would pay for the latency.

## 6. Analysis

**Why six microphones suffice.** Uniform DAS weights every microphone equally, so a badly placed one
actively hurts, and all 24 has two such defects: one recorder is *coplanar*, making directions above
and below its plane indistinguishable, and its closest pair sits 0.17 m apart, half a wavelength at
1 kHz, so over most of the band the map repeats and one source lights several directions (*spatial
aliasing*). The learned model recovers most of what uniform weighting discards, after which 18 more
microphones buy nothing measurable — and the baseline shows the same ordering, so this is a property
of the array, not of the network.

**Why temporal context looked useless and is not.** Comparing two separately trained models confounds
two effects of opposite sign: widening the input to five frames costs 1.6–6.2 pp before any history is
used, and history returns 2.6–6.8 pp. The small number in section 5.2 is that *net*, not an absence, and it
concentrates where section 5.3 predicts.

**Why detection fails while localization does not.** Both are data-limited, in different currencies,
and the contrast is the most transferable thing we learned. Localization is supervised by geometry:
every drone frame carries a direction from the GPS log, the beamformer turns the array into a picture
of where energy is, and 52 minutes of flight is tens of thousands of labelled directions. Detection
is supervised by *scenes*, and we have three. More ambient frames do not substitute for more
soundscapes: 24 channels of one frame are near-duplicates, pooling all of them scoring 0.662 against
one microphone's 0.681. Hence an 86k-parameter network at chance across sites while a 2 M-parameter
U-Net localizes to 5.5°, and hence a fix that is a microphone and an afternoon in ten places rather
than a bigger model.

## 7. Limitations and future directions

- **Beyond 90 m the drone is not in the beamformed map.** Two thirds of labelled positives there
  peak lower than a typical ambient frame, our error is almost all azimuth (53° against 11°), and
  early stopping on a pooled rate never selected for that bin.
- **The near field is our worst region and we do not know why** — 0–30 m gives 67% within 10°
  against 78% at 30–60 m, in every seed and array — and section 7.1 shows it is also what fails to transfer.
  Our steering vectors assume a plane wave, which a source metres from a 2 m array is not: a
  candidate, untested.
- **Three ablations aimed at known weaknesses returned nulls, and the nulls are informative.**
  Re-projecting the map onto a zenith disc, removing the pole distortion on the azimuth axis, changed
  nothing (71.0% ± 1.8 against 70.1% ± 3.6) and left the far bin worse; the sub-band count, our
  reconstruction of a detail the paper omits, moves the hit rate 0.5 pp across 8, 16 and 32 bands;
  and a log-frequency axis for the detector, motivated by a rotor's comb stretching multiplicatively
  with RPM, gained 0.02–0.06 AUROC against a fold spread of 0.06–0.22. Notes 31–33 keep the
  arguments, since each predicted something the measurement declined to show.
- **One drone, one dataset, two sites**, so the cross-airframe question is open, and settling it needs
  an outdoor, uncompressed recording of a comparable airframe — a data-collection task. The available
  alternative, indoor 64 kbps recordings of two Parrot toys, stacks airframe, environment, range and
  codec mismatches at once, and its no-drone class is ESC-50 itself, which would leak into section 5.5.
- **Untested:** the speed head from the unused GPS velocity labels, their Tversky loss and
  skip-attention, and K=9 on the other arrays, where section 5.3's curve says the next gain is.

### 7.1 Test 1: the sealed split, scored once

Three minutes from a different site and date, with its own array calibration, sealed since the first
commit and scored by one command after everything above was fixed.

*Median angular error in degrees by range over test 1's 1,592 drone frames, three seeds per U-Net
cell; one percentage point is 16 frames.*

| Test 1 | within 10° | 0–30 m | 60–90 m | 90–120 m | 120–180 m |
|---|---|---|---|---|---|
| DAS, 24 mics | **63.9%** | **5.9** | **3.7** | 36.3 | 58.7 |
| U-Net all24 K=1 | 52.2% ± 5.5 | 13.3 | 5.6 | 5.6 | 40.5 |
| U-Net dev3 K=5 | 56.7% ± 7.6 | 15.1 | 6.6 | **5.3** | 27.2 |
| U-Net dev3 K=9 | 52.0% ± 3.2 | 20.4 | 6.5 | 6.6 | **21.2** |
| *share of test-1 frames* | | *35%* | *30%* | *21%* | *4%* |

**The headline claim survives.** Six microphones on one recorder are not worse than 24 here either:
54.4% ± 4.6 against 52.2% ± 5.5 at K=1 and 56.7% against 53.2% at K=5 — the same direction on both
splits, on the cheaper hardware.

**Everything degrades and the seed spread quadruples**, from ±1.2 pp to ±5–8 pp, so no cell here is
separable from another and K=9's test-2 advantage is neither confirmed nor refuted pooled. The
*shape* section 5.3 predicted does survive, within `device 3`: K=9 gives its best far error (21.2° against
25.0° at K=1) and its worst near error (20.4° against 14.8°), deep history paying where evidence is
weakest and costing where it is strong. With 67 frames beyond 120 m that is suggestive, not
significant.

**Classical DAS wins pooled here, and the range breakdown says why.** Inside 90 m DAS is far better
at this site than at ours (5.9° and 3.7°, against a training-site baseline growing 8.6× over the same
span) and it still collapses beyond it, while the U-Net is flat at 5–6° out to 120 m. Two thirds of
test 1 sits inside 90 m, so the pooled number goes to DAS: the learned model's advantage is where it
always was, and what fails to transfer is its near-field behaviour — already our worst region on
test 2, and plausibly the most geometry-sensitive, this session having its own calibration. First
thing to investigate, and not something test 1 can be asked twice. **The detector reads 0.639 AUROC
and rejects 6% of negatives at 95% recall**: section 5.5 again on new ground, above chance and worth
nothing.

## 8. Contributions and reproduction

**[TO FILL: division of work]** — per-member contribution, as the course instructions require. Every
table here regenerates from committed code and each build step has a gate script printing `GATE PASS`
or `GATE FAIL`; `README.md` has the commands and `docs/design_notes.md` the full record, including
what did not fit.

## 9. References

B. J. Rodríguez et al., *Acoustic Imaging for UAV Detection*,
[arXiv:2508.00307](https://arxiv.org/abs/2508.00307), IWAENC 2026, and the same authors' *UAV
Acoustic Localization Dataset*, [DOI 10.26183/07nm-4p35](https://doi.org/10.26183/07nm-4p35), 2026.
Ronneberger, Fischer and Brox, *U-Net*, MICCAI 2015. Thiemann, Ito and Vincent, *DEMAND*, Zenodo
2013. Piczak, *ESC-50*, ACM Multimedia 2015.
