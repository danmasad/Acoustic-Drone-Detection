---
title: "Hearing Drones in 3D: What a Cheap Microphone Array Can and Cannot Do"
author: "Shai Livshits, Dan Masad"
date: ""
geometry: margin=0.8in
fontsize: 10pt
colorlinks: true
header-includes: |
  \usepackage{titlesec}
  \usepackage{titling}
  \usepackage{enumitem}
  \setlength{\droptitle}{-5em}
  \posttitle{\par\end{center}}
  \titlespacing*{\section}{0pt}{6pt}{2pt}
  \titlespacing*{\subsection}{0pt}{4pt}{1pt}
  \setlength{\parskip}{2pt}
  \setlist[itemize]{itemsep=0pt,topsep=2pt,parsep=0pt,leftmargin=1.2em}
---

**Category:** Applied — Audio / Convolutional Networks / Defense & Security.
**Code:** [github.com/danmasad/Acoustic-Drone-Detection](https://github.com/danmasad/Acoustic-Drone-Detection)

## 1. Motivation and problem definition

Drones are small, quiet targets that often fly without a radio link, which defeats cameras in
poor visibility and radio detectors outright. Sound is a cheap fallback: rotors are loud and
harmonically distinctive. From a ground microphone array we want a system that reports, ten
times per second, whether a drone is audible and in which direction — azimuth (compass bearing)
and elevation (angle above the horizon).

The practical obstacle is hardware. Published systems use dense arrays of 24 or more
synchronized microphones. If comparable accuracy were reachable with six microphones in one
off-the-shelf recorder, the cost of a deployable sensor would fall by roughly the same factor.
Our reference paper (Rodríguez et al., 2026) identifies temporal context as its own untested
future work, so our question was:

> **Can temporal context let a cheap 6-microphone array match a 24-microphone one?**

The answer turned out to be that the premise was wrong in an interesting way, which is what this
report is mostly about.

## 2. Related work

Classical sound-source localization uses delay-and-sum (DAS) beamforming, GCC-PHAT, SRP-PHAT,
MVDR or MUSIC to steer an array and read off the strongest direction. These are cheap and
interpretable but degrade under noise and at low signal-to-noise ratio, where sidelobes produce
phantom peaks indistinguishable from the source in a single frame.

Learned approaches usually regress direction directly from multichannel audio. Our reference
paper instead treats the beamformed energy map as an *image* and the localization task as
semantic segmentation, training a U-Net to output a spatial mask over the azimuth–elevation
grid and taking the centroid of the activated region as the direction estimate. This is the
formulation we adopt, because it makes the array a preprocessing choice rather than an
architectural one: the network sees a map, not channels, so the same network can be trained on
any microphone subset. That property is exactly what our experiment needs.

The paper reports its results against DAS on its own dataset, uses a single frame of input, and
evaluates detection only as a stability proxy on five minutes of no-drone audio. It released the
dataset and beamforming scripts but no training code or weights, so everything here is
reimplemented from the paper's description.

## 3. Data

The Western Sydney University UAV Acoustic Localization Dataset (CC BY-SA 4.0, 24 GB) contains
open-field recordings of a DJI Air 3 from a 24-microphone array — four independent recorders of
six microphones each, 48 kHz, with GPS flight logs at 10 Hz and per-recorder sample offsets for
alignment. Splits follow the paper: training is 52 minutes of flight across two sessions at
site 1, test 2 is 20 minutes at the same site, and test 1 is 3 minutes at a different site under
unseen conditions. **Test 1 was sealed** throughout development and model selection.

Ground truth comes from the flight logs: each 100 ms frame gets the drone's azimuth, elevation
and range, and row *k* of the 10 Hz log is frame *k* of the segment, which is the whole
alignment scheme. Labels for the segmentation task are binary discs of 10° radius around the
true direction, and drone presence is a per-frame binary label.

Negative (no-drone) audio is the dataset's weak point: the release labels roughly 5 minutes of
it. We recovered more by mining the pre-takeoff portion of every recording, which the dataset's
own splits never use, verifying with beamformed peak contrast that these windows are ambient
rather than drone, and discarding a 20 s margin before takeoff for motor spin-up. That took
training negatives from 4.8 to 8.2 minutes and from one scene to three, and gave test 1
negatives for the first time.

## 4. Method

**Features.** One pass over the audio caches the per-microphone short-time spectra of every
100 ms frame (381 bins over 200–4000 Hz, 3.4 GiB total). Everything downstream — microphone
subset, direction grid, sub-band count, frame stacking — is then a re-steer over this cache
rather than another read of 24 GB. Frequency-domain DAS turns a batch of cached spectra into
energy maps of 23 elevations × 91 azimuths at 4° resolution, averaged into 16 sub-bands over
200–2200 Hz. Beamforming per frequency and keeping the bands separate matters because a drone is
a harmonic stack: the same direction lights up in every band, while a sidelobe artefact does not.

**Model.** A U-Net (PyTorch) of depth 3 with 3×3 convolutions and two heads: a per-pixel mask
and a pooled presence logit. Two departures from the reference: azimuth uses circular padding,
since the map's left and right edges are the same bearing, and the presence head exists at all,
which is what makes real precision–recall measurable. Temporal context is K frames stacked on
the channel axis, current frame last, so the stack is causal and the label always belongs to the
most recent frame.

**Hyperparameters.** Adopted from the paper rather than re-searched, since re-searching per
condition would confound the comparison: depth 3, 3×3 kernels, Adam at lr 5e-3, class-weighted
cross-entropy. Two deviations, both measured: 32 base filters instead of 64 (2.3× faster *and*
better — 72% against 63% of frames within 10°), and a `LeakyReLU` presence head after the plain
ReLU version died in one seed. Early stopping monitors the 10° hit rate on test 2, patience 10,
minimum 15 and maximum 60 epochs, with the best checkpoint restored. Every condition uses the
identical protocol, which is what makes the cells comparable.

**Baseline.** Classical DAS with brightest-pixel decoding, scored with the same metrics as the
network, on the same frames.

**A separate detector.** The map-based presence head reads a beamformed map, so any external
audio injected for training would carry a spatial signature our rendering invented. We therefore
also built a per-channel detector: a 1-D CNN (86k parameters) over one microphone's 381-bin
spectrum, with frame decisions by averaging per-channel logits. It has no cross-channel
structure to exploit, so mono audio needs no rendering at all.

## 5. Experiments and results

All numbers are on test 2, with three seeds per cell in the headline grid. The pooled
within-cell seed spread is **±1.2 points**, which is the yardstick for every comparison below.

### 5.1 Does our pipeline reproduce the reference paper?

Their results are published as figures, so the comparison required adopting their conventions:
their distance bins, the mean rather than median error, and their detection rule, which counts
a frame as detected when the last three direction estimates agree within 10°.

| Test 2, 24 mics | 0–50 m | 50–100 m | 100–200 m | Stability false positives |
|---|---|---|---|---|
| Paper, DAS | 38° | 75° | 33° | 67.0% |
| **Ours, DAS** | **40°** | **72°** | **31°** | **68.6%** |

We reproduce their baseline within a few degrees in every bin, including their non-monotonicity
where 100–200 m scores better than 50–100 m. This is our strongest end-to-end check: it
exercises array geometry, per-recorder offsets, the STFT, the beamformer, label alignment and
the metrics at once, against an independent implementation.

### 5.2 Headline: microphone count and temporal context

| Array | Mics | Recorders | K=1 | K=5 | K=5 − K=1 |
|---|---|---|---|---|---|
| all 24 | 24 | 4 | 70.5% ± 1.5 | 70.1% ± 0.9 | −0.4 |
| **device 3** | **6** | **1** | **69.3% ± 0.7** | **71.1% ± 1.8** | +1.8 |
| best-6 (searched) | 6 | 2 | 52.1% ± 0.8 | 52.8% ± 1.2 | +0.6 |

Frames within 10°, mean ± sample standard deviation over three seeds.

**Six microphones on one recorder match twenty-four.** The gap is −1.2 points at K=1 and +1.0 at
K=5, both inside seed noise. **Temporal context does not reliably help**: +1.8 points on the
cheap array against a ±1.2 spread, nothing elsewhere, and median error unchanged.

### 5.3 Is the temporal mechanism real, even if the payoff is not?

Three interventions applied at test time to trained K=5 models, averaged over all nine runs:

| Intervention | Within 10° | Change |
|---|---|---|
| causal (unmodified) | 64.7% | — |
| freeze — current frame repeated K times | 60.4% | **−4.3** |
| shuffle — same frames, random order | 64.5% | −0.1 |
| reverse — time flipped | 64.5% | −0.2 |

The model does use the past: removing history costs 4.3 points. But it is insensitive to the
*order* of the past, which matches the physics — beyond about 70 m the true direction moves less
than one grid cell across a whole stack, so the usable cue is persistence, and persistence is
order-invariant. A drone *persists* while false peaks flicker.

### 5.4 The array search selected the worst array

| Array | Mics | Recorders | Sidelobe score | Within 10° |
|---|---|---|---|---|
| device 1+3 | 12 | 2 | −2.4 dB | 73.1% |
| all 24 | 24 | 4 | −3.6 dB | 70.5% |
| device 3 | 6 | 1 | −2.8 dB | 69.3% |
| symmetric-6 | 6 | 2 | −1.8 dB | 64.0% |
| device 1 | 6 | 1 | −1.5 dB | 59.0% |
| device 4 (coplanar) | 6 | 1 | −0.2 dB | 54.5% |
| quad-4 | 4 | 2 | −1.8 dB | 54.2% |
| **best-6 (searched)** | 6 | 2 | **−3.6 dB** | **52.1%** |

We had scored all 106,183 non-coplanar 6-microphone subsets on worst-case sidelobe level and
pinned the winner. It came last. The criterion correlates with accuracy at **−0.80 across every
array it did not select**, and including its own winner collapses that to −0.36.

### 5.5 Detection

| Detector | test 2 AUROC | Negatives rejected at 95% recall |
|---|---|---|
| Map-based presence head | 0.586–0.727 by array, seed sd 0.07–0.22 | 4–11% by cell, seed sd up to 8 |
| Per-channel spectral detector | 0.662, stable | 6.6% |

The map head is unusable: seed variance dwarfs every effect, and one run scored 0.288, ranking
no-drone frames *above* drone frames on the unseen scene. The per-channel detector ties it with
92× fewer parameters and does not swing.

**[TO FILL: ESC-50 confuser rejection]** — per-class rejection rates for the eight drone-like
classes, mixed into our own ambient audio. Code in `src/esc50.py`, not yet run.

## 6. Analysis

**Why six microphones suffice.** Uniform DAS gives every microphone equal weight, so a poorly
placed microphone actively hurts; the 24-microphone array includes a coplanar recorder with a
perfect front–back ambiguity and tight clusters that alias above 1 kHz. The learned model
recovers most of what uniform weighting throws away, and after that the extra 18 microphones buy
nothing measurable. The classical baseline shows the same ordering, which is what convinced us
it is a property of the array rather than of the network.

**Why the searched array failed.** This is a textbook failure of optimizing a proxy. The
beampattern was computed under a free-field model with exact geometry and perfect
synchronization; maximizing `true quality + model error` over 106,183 candidates selects hard on
the error term, so the winner of a large search is exactly where the proxy deserves least trust.
Every array we did *not* choose that way still obeys the criterion, which is what turns this from
an anecdote into a measurement. We initially blamed cross-recorder synchronization and tested
it: two of the three best arrays span two recorders, so that explanation is dead.

**Why temporal context does not pay.** The mechanism works — the network demonstrably reads the
past — but at these ranges the target barely moves between frames, so history mostly confirms
what the current frame already says. Context would be expected to help where single-frame
evidence is ambiguous and motion is large, which on this dataset is a small minority of frames.

**Why detection is weak.** It is a data limit, not a model limit. Eight minutes of negatives from
three scenes cannot support a detector that generalizes to a new location, and 24 channels of the
same frame are near-duplicates: averaging all 24 per-channel decisions (0.662) does no better
than one microphone (0.681). There is also a hard measurement floor — with the ambient audio in
the release, the finest false-alarm rate resolvable is 10.7 per hour on test 2, so any
operationally meaningful rate is below what this dataset can measure at all.

## 7. Limitations and future directions

- **Beyond 100 m we do not match the paper.** In that bin the drone's beamformed peak energy is
  *below* the mean on no-drone frames, our error is almost entirely azimuth (53° against 11° in
  elevation), and our early stopping never selected for it. Untested candidates: their polar
  reprojection, Tversky loss, and skip-attention, all of which we dropped.
- **One drone, one dataset, two sites.** Nothing here shows the result transfers to other
  airframes or environments.
- **Single seed** on the eight-array comparison, so gaps under about 5 points there are ties.
- **Detection needs more scenes**, not more architecture. The cheapest real improvement would be
  recording an hour of local ambient audio.

**[TO FILL: test 1 results]** — final numbers on the sealed unseen-site split, for the baseline,
the headline arrays at K=1 and K=5, and the detector.

**[TO FILL: secondary ablations]** — polar versus rectangular map, frequency-bin count, the speed
head from the unused GPS velocity labels, and K of 3 and 9.

## 8. Contributions and reproduction

**[TO FILL: division of work]** — per-member contribution, as the course instructions require.

The repository contains one gate script per build step, each printing `GATE PASS` or `GATE FAIL`,
and every table above regenerates from committed code:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
utils/download_dataset.sh && .venv/bin/python -m src.stft_cache   # 24 GB, then ~4 min
.venv/bin/python utils/detach.py logs/headline.log -- \
    .venv/bin/python -u utils/run_grid.py headline                # 18 runs, ~13.5 h
.venv/bin/python utils/summarize_grid.py                          # tables 5.2, 5.4, 5.3
.venv/bin/python utils/check_vs_paper.py                          # table 5.1
```

`docs/design_notes.md` records every measurement, decision and dead end in full, including the
ones that did not make this report.

## 9. References

1. B. J. Rodríguez, S. F. Chevtchenko, M. Herrera Martínez, Y. Bethi, S. Afshar. *Acoustic
   Imaging for UAV Detection: Dense Beamformed Energy Maps and U-Net SELD.*
   [arXiv:2508.00307](https://arxiv.org/abs/2508.00307), Aug 2025 (rev. Jul 2026); IWAENC 2026.
2. B. J. Rodríguez, S. F. Chevtchenko, S. Afshar. *UAV Acoustic Localization Dataset: 24-Channel
   Beamformed Recordings of a DJI Air 3 Drone with Synchronized GPS Flight Logs.* Western Sydney
   University, 2026. [DOI 10.26183/07nm-4p35](https://doi.org/10.26183/07nm-4p35).
3. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image
   Segmentation.* MICCAI, 2015.
4. S. Salehi, D. Erdogmus, A. Gholipour. *Tversky Loss Function for Image Segmentation.* MLMI,
   2017.
5. K. J. Piczak. *ESC-50: Dataset for Environmental Sound Classification.* ACM Multimedia, 2015.
6. V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015.
   (Frame stacking as channels.)
