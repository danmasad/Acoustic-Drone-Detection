---
title: "Hearing Drones in 3D: Convolutional Detection and Direction-Finding from a Reduced Microphone Array"
author: "Shai Livshits, Dan Masad"
date: "5 October 2026"
geometry: margin=0.62in
fontsize: 10pt
colorlinks: true
header-includes: |
  \usepackage{titlesec}
  \usepackage{titling}
  \usepackage{enumitem}
  \usepackage{caption}
  \usepackage{etoolbox}
  \usepackage{booktabs}
  \setlength{\droptitle}{-5em}
  \posttitle{\par\end{center}}
  \titlespacing*{\section}{0pt}{1.5pt}{0pt}
  \titlespacing*{\subsection}{0pt}{1pt}{0pt}
  \setlength{\parskip}{0.4pt}
  \linespread{0.93}
  \captionsetup{font=footnotesize,skip=1pt}
  \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3pt}}
  \setlength{\LTpre}{1pt}
  \setlength{\LTpost}{1pt}
  \setlength{\textfloatsep}{3pt}
  \setlength{\intextsep}{3pt}
  \setlist[itemize]{itemsep=0pt,topsep=1pt,parsep=0pt,leftmargin=1.1em}
---

**Project Category:** Applied — Audio / Convolutional Networks / Defense & Security.

**Code:** [github.com/danmasad/Acoustic-Drone-Detection](https://github.com/danmasad/Acoustic-Drone-Detection)

## 1. Motivation and problem definition

Drones are small, quiet targets that often fly without a radio link, which defeats cameras in poor visibility and radio detectors outright. Sound is a cheap fallback: rotors are loud and harmonically distinctive. From a ground array we want a report, ten times a second, of whether a drone is audible and in which direction. Published systems use 24 or more synchronized microphones; six in one off-the-shelf recorder would cost about four times less. Rodríguez et al. (2026) name temporal context as untested future work, so the question was whether a few frames of history could let six microphones match twenty-four.

They can, and the history is not why. Six microphones on one recorder match the full array on every drone frame. Extra frames help only where microphones are scarce. A kinematic filter after the network removes lobe hops. It does not produce the far-range accuracy the paper reports, and it does not fix the near field.

## 2. Related work

Classical localization steers the array and reads the strongest direction. Our baseline is **delay-and-sum**: undo the delays a candidate direction implies, sum, and take the energy. A correct guess adds in phase. The map over azimuth and elevation has a second failure mode, a **sidelobe**, whose delays line up almost as well as the true source. One frame cannot tell the phantom from the source, and a thinner array makes more of them.

The reference paper treats that map as an image and localization as segmentation: a U-Net predicts a mask, and the mask's centroid is the estimate. We adopt that because the array is then a preprocessing choice. The paper uses one frame, scores detection only by whether the direction sits still, and released data and beamforming scripts but no training code. Everything here is reimplemented from the description.

## 3. Data

The Western Sydney University UAV Acoustic Localization Dataset (CC BY-SA 4.0): a DJI Air 3 recorded on a 24-microphone array, four recorders of six, 48 kHz, with GPS logs at 10 Hz. Splits follow the paper. Training is 52 minutes at site 1. Test 2 is 20 minutes at the same site. Test 1 is 3 minutes at another site and was sealed through development. Row $k$ of a log is frame $k$ of the segment. Targets are 10° discs.

The release labels about five minutes of no-drone audio. We mined pre-takeoff audio the official splits never use, checked by beamformed contrast that those windows are ambient, and dropped 20 s for motor spin-up. Training negatives went from 4.8 to 8.2 minutes and from one scene to three. That is still not enough to learn a detector.

## 4. Method

One pass caches every frame's spectra (381 bins, 200–4000 Hz). Frequency-domain delay-and-sum then builds maps of 23 elevations by 91 azimuths at 4°, averaged into 16 sub-bands over 200–2200 Hz. The array is unambiguous only below 1.47 kHz, so the maps run past the alias-free band, and even all 24 microphones leave a 3.6 dB phantom beside the true peak.

The model is a depth-3 U-Net with 3×3 convolutions, circular padding on azimuth, a per-pixel mask, and a pooled presence head. Temporal context is $K$ frames stacked on the channel axis, newest last, so it is causal. Hyperparameters follow the paper where we could: Adam at $5\times10^{-3}$, class-weighted cross-entropy, early stopping on the test-2 hit rate with patience 10. Two measured deviations: 32 base filters rather than 64, which was both faster and better (72% against 63% within 10°), and a leaky presence head after a plain ReLU died in one seed. The baseline is brightest-pixel delay-and-sum on the same frames. A separate 1-D CNN (86k parameters) detects from one microphone, so it can be tested on audio we did not render.

The kinematic filter runs after the mask, on the decoded direction. It is not an input channel and not a reason to retrain. A DJI Air 3 in sport mode is published at 21 m/s, so a frame of 0.1 s allows 2.1 m of travel. At the GPS range that is a sky-angle cap, plus one 4° cell because the decoder sits on the grid. The stored direction updates at once when the new peak lies inside the cap, and only after five consecutive frames — half a second — when it does not. A hard hold with no second chance was tried; it wrecks the score, because some large jumps are the model correcting itself. A fixed 16° cap, the same rule with no range, landed within about a point of the range-dependent cap on the same-site 24-microphone run.

## 5. Experiments and results

The error on a frame is the angle between the predicted and true direction, each written as a unit vector. Azimuth and elevation are not scored apart: a compass flip overhead can be a small step on the sky. **Two summaries of that angle do not convert, and this report never puts them in one table.**

- **Against the paper.** Their summary only. The mean angle, their bins 0–50 / 50–100 / 100–200 m, and, on U-Net rows, their stability cut: the last three directions lie within 10°. Their delay-and-sum bars are every frame. Our row in that table uses the same cut.
- **Against our own variants.** Every drone frame, not the stable subset. The share whose sky angle is at most 10°, our bins 0–30 / 30–60 / 60–90 / 90–120 / 120–180 m, and the seed spread. Alongside that, the sideways miss $\mathrm{range}\times\sin(\mathrm{angle})$.

A mean is pulled by the tail. On the same-site flight, delay-and-sum past 100 m has a mean of 31°, next to their 33°, and 2% of those frames are inside 10°. Quoting one number as the other flatters whoever picked it. The new-site filter and the new-site paper bins below are a **second look** at dumps that had already been scored, not a new sealed result. "±" is the seed standard deviation, omitted when it rounds below 2. The pooled within-cell seed spread on the matched site is ±1.2 percentage points; a smaller gap is not a result. Three seeds cannot return a permutation $p$ below 0.10, so gaps across separately trained models are point estimates. The one comparison that survives block resampling is named below.

### 5.1 Against the paper, on their metric

Delay-and-sum on all 24 microphones reproduces their bars within a few degrees, including the non-monotonic far bin, and their stability rule calls 67.0% of no-drone frames detections against our 68.6%. That check exercises geometry, recorder offsets, the STFT, the beamformer, the labels, and the metric against an independent implementation.

*Mean angular error in degrees. U-Net rows are stable frames; delay-and-sum rows are every frame. The new-site block is a second look. Our matched-site U-Net range is three seeds (notes 23).*

| | Site | 0–50 m | 50–100 m | 100–200 m |
|---|---|---:|---:|---:|
| Paper, delay-and-sum | matched | 38° | 75° | 33° |
| Ours, delay-and-sum, 24 | matched | 40° | 72° | 31° |
| Paper, U-Net, stable | matched | 10° | 13° | 14° |
| Ours, U-Net 24, stable | matched | 7–10° | 9–12° | 31–74° |
| Paper, delay-and-sum | new, second look | 8° | 29° | 92° |
| Ours, delay-and-sum, 24 | new, second look | 9° | 23° | 83° |
| Paper, U-Net, stable | new, second look | 12° | 24° | 54° |
| Ours, 24 mics $K{=}1$, stable | new, second look | 16±4° | 10±3° | 20±9° |
| Ours, device 3 $K{=}1$, stable | new, second look | 14° | 12±3° | 11±2° |

**Where that leaves the comparison.** On the matched site, under 100 m, we are level with them and slightly ahead on the frames they keep (7–10° and 9–12° against 10° and 13°). Past 100 m there, their 14° still beats our stable-frame mean (31–74°). The filter does not change that sentence: it is not their cut, and on their metric the far bin on the matched site remains theirs. On the new site, second look, they lead under 50 m and we lead from 50 m out. Device 3 at one frame is 12±3° and 11±2° against their 24° and 54° on the frames they keep. That flight is 1,592 drone frames, about 2.7 minutes, with 238 frames past 100 m. It is a direction, not a second sealed trial.

### 5.2 Against ourselves, on every frame

*Share of drone frames inside 10°, matched site, our bins. Each entry is the raw decode, then the same frames after the filter. The remaining arrays, and the polar and Tversky grids, are in notes 47.*

| | 0–30 m | 30–60 m | 60–90 m | 90–120 m | 120–180 m | All |
|---|---:|---:|---:|---:|---:|---:|
| Delay-and-sum, 24 | 50→52 | 33→36 | 18→21 | 3→0 | 2→0 | 33→35 |
| 24 mics, $K{=}1$ | 66±5→66±5 | 79→83±3 | 73→82±6 | 57±3→74±12 | 27±11→55±13 | 70±2→76 |
| 24 mics, $K{=}5$ | 63→62 | 79±3→81±2 | 75→77±3 | 60±2→64±6 | 27±6→34±12 | 70→72 |
| Device 3, $K{=}1$ | 69±3→70±2 | 75→81±2 | 68→77 | 56→64±20 | 25±4→65±11 | 69→75±2 |
| Device 3, $K{=}9$ | 72±2→72±2 | 78±3→78±4 | 73±3→74±4 | 71±12→74±12 | 24±6→22±4 | 73→73±2 |

*Sideways miss for delay-and-sum on the matched site, our reading only. A 10° error is about 5 m at 30 m and 17 m at 100 m. The far-bin entry is that bin's 32° mean. Not a comparison with the paper.*

| | Under 30 m | Far bin |
|---|---:|---:|
| Typical sideways miss, 24 microphones | 3 m | 64 m |

**Six microphones match twenty-four.** At one frame the gap is 69.3% ± 0.7 against 70.5% ± 1.5, inside seed noise; at five frames it is still inside that noise. At nine frames, 0.9 s, device 3 reaches 73.2% ± 1.4 and all 24 stays at 70.1% ± 2.7. Depth pays only on the small array. The six a geometric search called optimal, out of 106,183 subsets, scores 52.1% ± 0.8. The proxy was over-fit: it predicts the arrays it did not select at −0.80, and including its own winner collapses that to −0.36 (notes 22).

**The filter, on this reading.** Same site, every frame: 24 microphones at one frame go from 70% to 76% inside 10°, and device 3 at one frame from 69% to 75%. Once five or nine frames are already stacked it adds about a point, and device 3 at $K{=}9$ stays at 73%. The gain is in the far bins. Inside 30 m the cells do not move. On the new site, second look, the same one-frame filter goes from 52% to 61% (24 microphones) and from 54% to 61% (device 3), and device 3 at $K{=}9$ stays at 52%. The single-frame network plus the filter is the combination that moves; the stack learned the same smoothness on the training site and did not carry it. Where the raw track is stably wrong, the filter holds the error. It is not a repair for a bad array (notes 47).

`utils/analysis/score_kinematic_filter.py` regenerates the filter and the sideways-miss numbers.

History measured on weights that are already trained agrees with the table. Corrupting the past of the $K{=}5$ models costs 4.3 percentage points, in shrinking steps, and the climb ends near 0.8 s. Deleting history from the $K{=}9$ weights costs 5.1 points, block-resampled, Holm-adjusted $p = 0.0007$: about +0.5 inside 30 m, +8.7 at 60–90 m, and +21.8 at 90–120 m. Order does not matter. The cue is persistence. Lobe hops are what flicker, and they are what the filter removes. The smooth miss inside 30 m is the near field, and the filter does not fix it.

**Detection fails across scenes.** Held out one scene at a time, the per-channel detector scores 0.488 ± 0.056 AUROC. The 0.662 measured earlier was the recording: training and test ambient were two halves of one afternoon. Time and extra ambient both sharpen the training site and do not travel; the false-alarm wall on other soundscapes is in notes 30 and 38–39. Past about 90 m the drone is often not the loudest peak in the map, so neither detector nor localizer has a lock, and two thirds of labelled positives there are quieter than typical ambient. On the sealed new site the presence score reads 0.639 AUROC.

## 6. Analysis

Uniform delay-and-sum weights a badly placed microphone as heavily as a good one. All 24 has a coplanar recorder and a pair 0.17 m apart, so the map repeats over much of the band. After the network learns to discount that, eighteen further microphones buy nothing measurable, and the baseline has the same ordering. The 3.6 dB phantom is the ceiling those extra channels do not remove.

Comparing two trained models confounds effects of opposite sign. Widening the input costs accuracy before any history is used, and history earns that cost back, mostly past 60 m. Section 5.2's small $K{=}5$ gap is that net.

Localization is supervised by geometry: every drone frame carries a direction. Detection is supervised by scenes, and we have three. More frames of one field are near-duplicates. An 86k-parameter network at chance across sites, beside a U-Net that localizes, is a data fact. The fix is another afternoon in another place.

## 7. Limitations

- **Past about 90 m** the labelled drone is often quieter than ambient in the beamformed map. Early stopping on a pooled hit rate never selected for that bin. On the matched site, their stable-frame 14° remains ahead of us there.
- **Inside 30 m** the learned cells sit below the next bin, and the filter leaves them there. On the new site, second look, the closest 30 m stay near 30% inside 10°. Steering assumes a plane wave, which a source metres from a 2 m array is not. Untested.
- **Polar discs, Tversky loss, and the rest of the array list** were measured. None changes the claim above; the grids are in notes 32, 36, and 47. Sub-band count moves the hit rate by about half a point.
- **One drone, one dataset, two sites.** The sealed new-site score, once, already said device 3 is not worse than all 24 (54.4% ± 4.6 against 52.2% ± 5.5 at one frame) while the seed spread widens to ±5–8 points and delay-and-sum wins the pool, because most of that flight is inside 90 m. The paper bins and the filter on that flight, above, are a second look at those dumps.

## 8. Contributions and reproduction

**Dan Masad** framed the cheap-array question, wrote the pipeline the tables run on (dataset, cache, delay-and-sum, U-Net, detector, metrics), ran the experiments those tables quote, including the one sealed test-1 score, and wrote the original sections 1–7. **Shai Livshits** wrote the four-phase review and notes 43–47: the no-retrain error analyses, a new baseline under a true test-2 split and leave-one-flight-out, and the kinematic filter with the two readings in section 5. No ablation beat that baseline by more than seed spread. The new-site rows marked as a second look reuse the dumps from that sealed score. Sealed tables regenerate from the `GATE PASS` scripts in `README.md`. The filter and the sideways-miss numbers regenerate from `utils/analysis/score_kinematic_filter.py`. `docs/design_notes.md` is the lab record.

## 9. References

B. J. Rodríguez et al., *Acoustic Imaging for UAV Detection*, [arXiv:2508.00307](https://arxiv.org/abs/2508.00307), IWAENC 2026, with its [dataset](https://doi.org/10.26183/07nm-4p35). Ronneberger et al., *U-Net*, MICCAI 2015. Thiemann et al., *DEMAND*, 2013. Piczak, *ESC-50*, 2015.
