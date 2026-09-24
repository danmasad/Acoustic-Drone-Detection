# Review plan — four-phase walkthrough

Working document from a structured review of the pipeline (24 Sep 2026), phase by phase.
Each phase records what the code does today, what was questioned, and the improvement plan
that came out of it. Nothing here has been implemented yet unless marked **done**.

Protocol for anything that retrains: develop on leave-one-flight-out over the three site-1
flights, never on test 1. Test 1 has been scored once; a second score needs a prediction
registered in `docs/prereg_test1.md` first, and a predicted gain of at least ~9 pp, which is
the smallest difference three seeds can resolve on that split.

---

## Phase 1 — raw signal preprocessing and representation

### What the code does (`src/stft_cache.py`, `src/beamform.py`, `src/data.py`)

- 24 channels from four hand-started recorders, aligned by the dataset's integer
  `corrections_samples`. Within a recorder the six channels share one clock.
- 5th-order Butterworth 200–8000 Hz, `filtfilt`, inherited from `create_dataset.py`.
- 100 ms frames (4800 samples), non-overlapping, **rectangular window** (plain `rfft` of the
  raw slice). 10 Hz bin spacing; 381 bins in 200–4000 Hz cached as complex64.
- Frequency-domain DAS over the 23 × 91 grid, energy in **decibels**, averaged into 16
  sub-bands over 200–2200 Hz.
- Input standardization: **one global mean/std** from training batches (`input_stats`),
  chosen deliberately so per-frame level (the detection cue) is preserved.

### Findings

1. The rectangular window is inherited, not chosen, and its leakage was never tested. Its
   sidelobes (−13 dB, 6 dB/octave) smear the rotor harmonic comb across sub-band boundaries
   and let sub-200 Hz wind leak upward. That leakage is the only live reason the Butterworth
   filter exists; its 8 kHz upper cutoff does nothing the bin selection does not.
2. Recorder offsets are trusted from the JSON. Two-recorder subsets perform well on site 1, so
   they are not badly wrong there, but test 1 has its own corrections and was never checked.
3. No spectrogram of the drone has been looked at. Whether the blade-pass comb is in-band, how
   many harmonics, and whether it survives to 90 m are all unknown — and bear directly on the
   far-field failure and on what the detector is (not) learning.
4. Global dB standardization from site 1 is a plausible mechanism for the 15–18 pp drop on the
   unseen site: a different gain or ambient level shifts every input pixel. Detection has since
   moved to a separate model, so the reason for keeping level in the map no longer applies to
   localization.

### Improvement plan

| # | Action | Cost | Settles |
|---|---|---|---|
| 1 | Spectrogram figure: drone at ~10, 50, 100 m and a no-drone segment, 0–2.5 kHz, 10 Hz bins, blade-pass harmonics marked | 30 min, no training | Is the comb in-band, does it survive range, is there a cue the energy map discards. Report figure |
| 2 | Alignment check on test 1: cross-correlate one channel per recorder pair after corrections, plot the lag of the peak | 30 min, no training | Rules alignment error in or out as a cause of the test-1 drop |
| 3 | Per-frame standardization of the dB maps in `to_maps` (mean-subtract, std-divide per map); retrain `device3` K=1 and K=9, 3 seeds, leave-one-flight-out. Keep frame mean dB as one scalar side input if the presence head needs it | ~4 h compute | Is the cross-site drop a level shift. Best single bet for transfer |
| 4 | Hann-window cache, same frame boundaries, no overlap; rerun the DAS baseline only | ~10 min | Does leakage matter. Unchanged → keep rectangular and document; improved → retrain headline cells |
| 5 | Document the filter honestly: inherited, only live effect is pre-FFT wind suppression; drop it if 4 adopts Hann | 10 min | Removes an implied claim |
| 6 | Frame-length paragraph: 100 ms ↔ 10 Hz label rate, <1 grid cell of motion per frame beyond ~30 m, 10 Hz bins resolve the comb | 10 min | Turns an inherited parameter into a reasoned one |

Priority if only two: 1 and 3.

---

## Phase 2 — feature extraction and dataset preparation

### What the code does (`src/data.py`, `src/labels.py`, `src/esc50.py`)

- Labels: GPS row *k* ↔ audio frame *k*, no propagation-delay shift. Array heading is one
  hand-measured `initial_azimuth_deg` per session, applied as a constant offset, unchecked.
- Target: 10° disc on the sphere rasterized to 23 × 91; ~2% of pixels positive on a drone
  frame. `has_drone` is all-ones for a flight segment — every frame is a positive target,
  including the two thirds of frames past 90 m whose beamformed peak is below ambient.
- Batching: global shuffle, bucketed by session so one steering tensor per batch; K-frame
  stacks are causal and never cross a segment boundary.
- Class balance: mask positives weighted `1/pos_rate` (cap 100); presence head weighted by
  inverse drone rate (92% of training frames are drone).
- **Augmentation of the map model: none.** Detector only: ESC-50 confusers mixed at a
  per-frame energy ratio (−5 to +20 dB, clean of split statistics); DEMAND ambient
  level-matched to the *training* no-drone median (fixed from an earlier test-2 leak).
- `speed_mps` label computed, unused.

### Findings

1. Heading calibration is probably right (DAS median azimuth error on test 1 is 2.9°) but is
   never checked; the check is mean *signed* azimuth error per session.
2. Range coverage: test 2 has 34% of drone frames under 30 m, test 1 35%. Training
   distribution unknown — needs the flight CSVs (dataset not on this machine). Decides
   whether the near-field failure is coverage or physics.
3. Label lag is uncorrected: sound from 100 m arrives 3 frames after the GPS fix. Systematic,
   in the direction of travel, largest where the model is weakest.
4. Inaudible far-field positives are trained on as if audible.
5. Level-matching is clean of test statistics; should be asserted in a gate.
6. No augmentation is the largest gap. The transfer failure is site-1 → site-2, and
   augmentation is the standard tool for invariance to the nuisances that differ between
   sites (gain, calibration, bearing coverage, SNR).

### Improvement plan

| # | Action | Cost | Settles |
|---|---|---|---|
| 1 | Shift GPS row by `round(range_m/343 × 10)` frames in `labels.py` | 15 min + retrain | Removes a range-dependent label bias |
| 2 | Histogram of train / test-2 / test-1 drone frames by range bin and by azimuth; mean signed DAS azimuth error per session | 30 min, needs dataset | Coverage gaps; heading calibration |
| 3 | Map-model augmentation on complex spectra before beamforming: per-channel gain jitter (±3 dB), channel dropout (p≈0.1), additive complex noise at random SNR (−10 to +20 dB), azimuth roll by multiples of 4° with the target rotated | 1–2 h code, retrain headline cells | Site invariance; the core transfer lever |
| 4 | Exclude or down-weight far frames with DAS peak contrast at chance from the *training* loss; keep them in evaluation, reported separately | 30 min + retrain | Stops training on labels the audio cannot support; may change what early stopping selects |
| 5 | Oversample / weight near-field frames if item 2 shows they are rare in training | 15 min + retrain | Near field as coverage |
| 6 | Cepstral / harmonic-comb (scale-free) feature for the detector, under leave-one-scene-out | 2 h | Whether detection fails on the feature or on the data |
| 7 | Gate: assert no test-split statistic feeds any training-time transform | 15 min | Leakage guard |

Priority: 3, then 1 and 4 (correctness), then 2 as soon as the dataset is available.

---

## Phase 3 — model architecture and training pipeline

### What the code does (`src/models.py`, `src/train.py`)

- Depth-3 U-Net, base 32 (paper 64), 3×3 conv → BatchNorm → ReLU ×2 per block, 2×2
  max-pool, nearest upsampling with skip concatenation, circular padding in azimuth. Mask head
  after `Dropout2d(0.1)`; presence head on avg+max pooled bottleneck. ~2 M parameters.
- Loss: BCE on the mask with `pos_weight = 1/pixel_pos_rate` (≈40, cap 100) + BCE on presence
  weighted by inverse drone rate. Tversky available.
- Adam, **constant lr 5e-3**, batch 32, no weight decay, no clipping, no schedule.
- Model selection: best `hit_10deg` on **test 2** per epoch, patience 10, min 15, max 60.
  No validation split.

### What the committed training histories show (`runs/*/metrics.json`)

Identical pattern in every run inspected (`device3` K=1/K=5, `all24` K=1, seed 0):

- Train loss 0.74 → 0.05 while test-2 loss 0.72 → 8.5: severe overfitting in the loss; hit
  rate stays flat because argmax only needs the pixel ranking, so the model grows
  overconfident rather than wrong.
- `hit_10deg` on test 2 swings ±5 pp epoch to epoch (59–70%). Best-of-~25 checkpoint
  selection on that metric is selection bias: test-2 numbers are optimistic by a few pp;
  between-cell comparisons survive because the bias is shared.
- Training-loss spikes (0.08 → 0.23, 0.06 → 0.16) at constant lr 5e-3: lr too high for the
  base-32 model without decay.
- BatchNorm running statistics come from site-1, session-bucketed batches; at test 1 the
  network normalizes site-2 activations with site-1 statistics — a second place (after input
  standardization) where the training site's level is baked in.

### Improvement plan

| # | Action | Cost | Settles |
|---|---|---|---|
| 1 | Validation split from training flights (leave-one-flight-out, or interleaved 30 s blocks); early-stop on it; test 2 becomes a true test | protocol change, retrain | Selection bias on every test-2 number |
| 2 | LR schedule (`ReduceLROnPlateau` on val loss, or cosine over 30 epochs) + gradient clipping at norm 1.0 | 10 min + retrain | Epoch-to-epoch noise; loss spikes |
| 3 | `AdamW`, weight decay 1e-3; dropout 0.2 inside encoder blocks | 10 min + retrain | Overfitting |
| 4 | Average weights of the last 5 epochs (or 3 best by validation) | 20 min | Lucky-epoch variance |
| 5 | `pos_weight` cap ≈10, or soft Gaussian target instead of hard 10° disc; re-test plain Dice (α=β=0.5) | 30 min + retrain | Calibration; centroid accuracy |
| 6 | GroupNorm instead of BatchNorm, or recompute BN statistics on the target session's unlabelled inputs before scoring | 20 min + retrain | Site-1 statistics inside the network |
| 7 | Reduce `presence_weight` on the map model; head is diagnostic only now that detection lives in `src/detector.py` | 2 min | Dead head competing for gradient |

Architecture itself is appropriate and stays. Detector architecture is appropriate; its
failure is data/feature (Phase 2), not model.

Execution order: one new baseline carrying all protocol/correctness fixes together
(items 1, 2, 3, 7 here; Phase 1 §3; Phase 2 §1), then ablate the transfer-specific changes
(Phase 2 §3 augmentation; item 6 here) one at a time against it.

Closed questions: the planned three-learning-rate sanity check **never ran** — every run on
both branches uses lr 5e-3. Dropout exists only before the mask head and in the presence
head, none in the encoder.

---

## Phase 4 — evaluation, error analysis and improvements

### What the code does (`src/metrics.py`, `results/`)

- Detection: AUROC (tie-aware rank-sum), AUPRC, specificity at 95% recall, best-F1 operating
  point, false alarms/hour at that threshold. Scores are raw logits. AUROC is the summary
  because test 2 is 81% positive. Correct choices, correctly justified.
- Localization: great-circle error between argmax cell and GPS direction on drone frames
  only; median, mean, az/el medians, hit rate at 5/10/20°, median per range bin. Headline
  tables use the grid-snapped argmax; centroid decoding exists.
- Error analysis already done on Dan's branch: by range, axis, history depth, scene (LOSO),
  frame-paired K comparisons, cluster-robust significance with Holm correction,
  pre-registration, sealed test set scored once.

### Findings

- Metrics are appropriate and well implemented; nothing to replace.
- Missing, all standard and cheap: a confusion matrix at a stated operating point; a
  calibration (reliability) curve — the val-loss explosion says probabilities are
  miscalibrated even where ranking is right; predicted-vs-true azimuth scatter and error
  vs. **elevation** (the disc's 50–100 m gain hints at an elevation effect, not range);
  cluster-robust CIs on the headline cells themselves.
- Failure modes tied to evidence: (1) near field, worst on both splits, fails to transfer —
  plane-wave steering error ≈ D²/(2rc) ≈ 0.25 ms at 10 m, a quarter-period at 1 kHz;
  (2) far field, drone below ambient in the map — an SNR limit, answered by integration;
  (3) detection learns soundscapes — needs scale-free features or a geometry-based score;
  (4) optimistic test-2 numbers from checkpoint selection; (5) ±5–8 pp seed spread on test 1.

### Improvement plan

| # | Action | Needs data? | Cost | Settles |
|---|---|---|---|---|
| 1 | Confusion matrix at the 95%-recall threshold (test 2, LOSO folds, test 1) | no | 30 min | Detection failure made legible |
| 2 | Reliability diagram for presence and mask probabilities | no | 30 min | Miscalibration; justifies Phase 3 loss changes |
| 3 | Predicted-vs-true azimuth scatter; error vs. elevation; per range bin | no | 1 h | Random vs. systematic far-field error; disc gain as an elevation effect |
| 4 | Cluster-robust CI on each headline cell's hit rate | no | 30 min | Intervals on the headline table |
| 5 | **Localizer persistence as a detection score** (direction variance over the last *n* frames), AUROC under LOSO | no | 2 h | Whether geometry-based detection transfers where timbre does not |
| 6 | **Hybrid DAS/U-Net decoder** gated on DAS peak contrast; threshold set on test 2 only | no | 2 h | Near-field recovery without retraining |
| 7 | Seed ensemble: average the three committed checkpoints' heatmaps before decoding (predict pass that dumps heatmaps) | yes | 1 h | Variance reduction; better point estimate |
| 8 | Near-field (spherical) steering at a fixed reference range as extra input channels | yes | 2 h + retrain | The plane-wave suspect |
| 9 | Report: detection section around the confusion matrix and LOSO; far bin as "beyond acoustic range"; state test-2 selection explicitly until Phase 3 §1 lands | no | writing | Honest framing |

Items 1–6 run today from `results/preds/`. Items 5 and 6 are the two with a real chance of
changing the headline story.

---

## Consolidated execution order

**Track A — no dataset needed, start now.**
Phase 4 §1–6; Phase 1 §5–6 (writing); Phase 2 §7 (gate); Phase 3 §4 (checkpoint averaging
code, tested on committed `best.pt` files where more than one epoch is available).

**Track B — download the dataset (~54 min) and rebuild the cache (~4 min) in the background
while Track A runs.** Then, in order:
1. Phase 1 §1–2 and Phase 2 §2 (spectrogram, alignment check, range/azimuth histograms,
   signed azimuth error per session) — measurements, no training.
2. Phase 1 §4 (Hann cache + DAS baseline) — decides the window before any retrain.
3. **New baseline**: Phase 3 §1 (validation split) + §2 (schedule, clipping) + §3 (AdamW,
   encoder dropout) + §7; Phase 1 §3 (per-frame standardization); Phase 2 §1 (label lag) +
   §4 (far-frame weighting). `device3` K=1 and K=9, `all24` K=1, 3 seeds each, ≈9 runs.
4. Ablate against the new baseline, one at a time: Phase 2 §3 (augmentation), Phase 3 §6
   (GroupNorm / BN re-estimation), Phase 3 §5 (soft target / Dice), Phase 4 §8 (near-field
   steering).
5. Phase 4 §7 (seed ensemble) on whatever wins.

**Test 1** stays untouched through all of this. If step 4 produces a cell that beats the new
baseline by more than seed spread under leave-one-flight-out, register the prediction in
`docs/prereg_test1.md` with an expected gain ≥ 9 pp and score test 1 once.

---

## Status

| Item | State | Where |
|---|---|---|
| Phase 4 §1 confusion matrix | done | `utils/confusion.py`, `results/confusion.json`, notes 43 |
| Phase 4 §2 presence reliability | done (mask pixels need heatmaps: Track B) | `utils/calibration.py`, `figures/calibration_presence_test_2.png`, notes 43 |
| Phase 4 §3 error structure | done | `utils/error_structure.py`, `results/error_by_elevation.json`, `figures/error_structure_test_2.png`, notes 44 |
| Phase 4 §4 per-cell intervals | done (`all24` K=9 needs a dump) | `utils/significance.py` section 6, `cell_ci` in `results/significance.json`, notes 40 addendum |
| Phase 4 §5 persistence detector | done on test 2; cross-scene test is a test-1 prereg candidate | `utils/persistence_detector.py`, `results/persistence_detector.json`, notes 45 |
| Phase 4 §6 hybrid decoder | done: null (oracle ceiling +1.7–2.7 pp); located an `all24`-only near-field failure; B1 lag check and §8 steering tell its two candidate causes apart | `utils/hybrid_decoder.py`, `results/hybrid_decoder.json`, notes 46 |
| Phase 1 §5–6 filter/window and frame length | done (writing; frame motion measured from test-2 labels) | notes 11, "Two inherited parameters" |
| README | done | review plan in the document table, new scripts, Track B under "What is left" |
| Phase 3 protocol flags (B3) | code done, every flag defaults off; the no-data gate passes (run ids, 24 checkpoints, loss, maps, weights, label lag). The epoch-0 check waits on the cache | `src/train.py` `PROTOCOL`, `utils/check_protocol_flags.py [--with-data]` |
| Phase 3 ablation flags (B5) | code done, default off, checked in the same gate (Dice, GroupNorm, soft target, azimuth roll, identity augmentation, near-field limit, run ids). The runs wait on B4 | `--augment`, `--norm-layer group`, `--target soft`, `--loss dice`, `--nearfield-m`; `src/augment.py` |
| Seed ensemble (B6) | tooling done, self-test passes; numbers wait on the cache | `python -m src.predict --heatmaps`, `utils/ensemble.py` → `results/ensemble.json` |
| Checkpoint averaging (B7) | tooling done, self-test passes; needs B4 runs trained with `--save-last` | `python -m src.train --save-last N`, `utils/average_checkpoints.py [--gate]` → `results/checkpoint_averaging.json` |
| Phase 4 §2 mask-pixel reliability (B8) | tooling done, self-test passes; needs heatmaps | `utils/mask_calibration.py` → `results/calibration_mask.json`, `figures/calibration_mask_test_2.png` |
