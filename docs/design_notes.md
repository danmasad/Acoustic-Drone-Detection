# Design Notes — Acoustic UAV Detection & Localization

Working notes behind `proposal_uav_acoustic_seld.md`. Everything here is either a
verified fact about the dataset or a decision we deliberately deferred out of the
proposal. Kept separate so the proposal stays short and non-committal about choices we
should make only after seeing how the implementation behaves.

Distinction used throughout: **inherited** = fixed by the dataset or its published
code, so quoting it commits us to nothing. **Ours** = a design choice we are free to
revise.

---

## 1. Numbers inherited from the dataset

Source: `code/dataset_creation/create_dataset.py` in the dataset release.

```python
EXPERIMENT_PARAMS = {
    "CHANNELS":          6,
    "sample_rate":       48000,
    "chunk_duration_s":  0.1,
    "speed_of_sound":    343,
    "lowcut":            200.0,
    "highcut":           8000.0,
    "azimuth_range":     np.arange(-180, 181, 4),   # 91 values
    "elevation_range":   np.arange(0, 91, 4),       # 23 values, tops out at 88 deg
    "angular_threshold": 10,                        # label radius in degrees
}
```

| Quantity | Value | Notes |
|---|---|---|
| Direction grid | 91 azimuth x 23 elevation = 2093 cells | 4 deg steps; verified by re-running the beamformer |
| Elevation coverage | 0 to 88 deg | No true zenith cell — overhead drone lands in the top row |
| Frame length | 100 ms | Matches the GPS log rate, so one frame = one label |
| Label tolerance | 10 deg | Cells within 10 deg of true direction get label 1 |
| Sample rate | 48 kHz, 32-bit PCM | 6 channels per WAV file, 4 files per session |
| Bandpass in script | 200–8000 Hz | Array's *useful* band is 200–4000 Hz (aperture limit) |
| Paper's model band | 200–2200 Hz, 16 FFT bins | Narrower than the script's filter |

Why 4 deg steps: roughly the array's real angular resolution given 4 cm–1.1 m
microphone spacing over 200–4000 Hz. Finer steps would interpolate blur, not add
information.

## 2. Microphone array

24 channels = **four synchronized Zoom F6 recorders, 6 channels each**. Geometry is
three tripod legs (6 mics each, upright tetrahedral) plus one horizontal ring of 6.
Sync via a synthetic impulse at the array origin, accurate to ±1 sample at 48 kHz
(≈7 mm of acoustic error).

| Ring | Mics | Radius (m) | Height (m) | Azimuth offsets |
|------|------|-----------|-----------|-----------------|
| Z3 | 1–3 | 0.10 | 1.12 | 0, −120, −240 |
| Z3 | 4–6 | 0.17 | 0.92 | 0, −120, −240 |
| Z2 | 7–9 | 0.25 | 0.77 | 0, −120, −240 |
| Z2 | 10–12 | 0.32 | 0.60 | 0, −120, −240 |
| Z1 | 13–15 | 0.42 | 0.42 | 0, −120, −240 |
| Z1 | 16–18 | 0.63 | 0.02 | 0, −120, −240 |
| Z0 | 19–24 | 0.25 | 0.77 | −40, −80, −160, −200, −280, −320 |

Note: `alignment_params.json` carries a **per-session** `mic_array` block, because the
array was recalibrated between sessions. Always read geometry from the session's own
JSON, never from the script's `DEFAULT_MIC_ARRAY` fallback.

### Aperture of every subset that costs nothing to try

Computed from the Oct_11_2024 geometry. Channel order in the concatenated array is
device order, so **each recorder is already a 6-mic subset** — no rewiring, no
re-derivation, just column selection.

| Subset | Channels | Aperture | Min spacing | Rows in table above |
|---|---|---|---|---|
| All 24 | 0–23 | **1.30 m** | 0.17 m | all |
| Device 1 | 0–5 | 0.31 m | 0.17 m | Z3 (r = 0.10, 0.17) |
| Device 2 | 6–11 | 0.55 m | 0.18 m | Z2 (r = 0.25, 0.32) |
| Device 3 | 12–17 | **1.09 m** | 0.45 m | Z1 (r = 0.42, 0.63) |
| Device 4 | 18–23 | 0.49 m | 0.17 m | Z0 ring |
| Greedy-spread 6 | 2, 9, 11, 15, 16, 17 | **1.30 m** | 0.55 m | mixed |

Three consequences, and they restructured the whole proposal:

1. **Six microphones can match the full array's aperture exactly** (1.30 m). Beamwidth
   scales as λ/aperture, so the greedy-spread 6 has the *same* angular resolution as all
   24 — 30° at 500 Hz, 15° at 1 kHz, 6.9° at 2.2 kHz. The other 18 microphones buy
   **array gain (≈6 dB) and sidelobe/grating-lobe rejection, not resolution.**
2. **The array is already spatially aliased.** Alias-free beamforming needs spacing
   < λ/2 = 7.8 cm at 2.2 kHz; the full array's minimum spacing is 17 cm, and
   greedy-spread 6 is 55 cm. So sparse subsets produce genuinely ambiguous maps with
   false peaks, not merely blurrier ones. This is exactly the failure a per-pixel CNN
   with temporal context might undo, and argmax cannot.
3. **Device 1 vs Device 3 is a free controlled experiment:** same count (6), same
   hardware, same session, aperture 0.31 m vs 1.09 m — a 3.5× difference. Count fixed,
   spread varied. Device 3 alone reaches 84% of the full aperture with **zero
   inter-device synchronization**, which is the cheapest realistic build.

## 3. Splits (as the authors defined them)

| Split | Content | Duration |
|---|---|---|
| Train | `training/Nov_25_2024/flight_11` + `training/Mar_18_2025/flight_1` + `test_2/.../flight_2` segment `no_drone_1` | 22 + 30 + ~4.8 min |
| Test 1 | `test_1/Oct_11_2024/flight_5` — **unseen site** | 3 + 1 min |
| Test 2 | `test_2/Mar_18_2025/flight_2` segments `drone` + `no_drone_2` — matched site | 20 + ~4.8 min |

Trap: **all** no-drone audio lives in the tail of one continuous recording
(`flight_2`, seconds 1290–1864), split into `no_drone_1` (train) and `no_drone_2`
(test) so frames are never shared. Do not pre-cut the WAVs; use `skip_seconds` /
`skip_seconds_end` from the JSON as `create_dataset.py` does. Also `flight_2` has
`*_part2.wav` files that were **not** used in the paper.

## 4. Logistics, measured

- License CC BY-SA 4.0. Open HTTP, no registration, HTTP range requests supported.
- 46 GB total; ~27 GB excluding the 360° reference video (~19 GB) which we don't need.
- Throughput measured: ~1 MB/s single connection, ~1.6–2 MB/s per connection with two
  in parallel. Full audio pull is an overnight job. Start it on day one.
- `test_1` alone is 1.1 GB (~20 min) — use it to build and debug the pipeline.

## 5. Compute: why the official pipeline is unusable, and our fix

`create_dataset.py` stores the full beamformed **waveform** at all 2093 directions per
frame: `beamformed_data` shape `(91, 23, 4800)` float32 = **40 MB per 100 ms frame**,
which is why the README quotes ~2 TB of derived data.

Our fix: steer in the frequency domain and keep only energy per
direction × frequency bin. Measured on a real slice:

| | Official (time-domain, full waveform) | Ours (frequency-domain, energy only) |
|---|---|---|
| Per frame | 40 MB | 0.13 MB (16 bins) |
| Whole dataset | ~2 TB | a few GB |
| Speed | numba, integer sample delays | 188 frames/s on one CPU core |

Fractional delays come free from the phase ramp, so it is *more* accurate than integer
sample shifting, not a shortcut. Nothing is lost for modelling, since the paper's own
network consumes energy maps anyway.

## 6. Reference paper (arXiv:2508.00307)

Their model input is **46 × 46 × 16** ("2E × 2E × F", E = 23) — the 91 × 23 grid
remapped onto a **polar disc**. We plan to use the plain rectangular 91 × 23 map
because it's simpler. **This is a divergence, not a match.** See section 7 for the full
argument and what it would cost to follow them.

Their hyperparameters (found via Optuna): 16 FFT bins, 64 base filters, depth 3, 3×3
kernels, lr 0.005, skip-attention, Tversky loss. Model selection was done on Test 2;
Test 1 was held out entirely.

Their reported results:

| | False-positive rate (no-drone) | DoA error |
|---|---|---|
| DAS beamforming | 67.0% | grows past ~90 m |
| Their U-Net | 14.9% | lower across all distance bins |
| On DCASE 2019 TAU | — | 68% F1 @ 4.9° (multi-source), 79% F1 @ 3.0° (single) |

Gaps we can own, all verified in their text: the model is strictly **single-frame** and
they name temporal context as future work; detection is evaluated only as a five-minute
false-positive proxy, with no precision/recall or false-alarm rate; the GPS
**velocity** labels (`vx/vy/vz`) ship with the data but go unused; they claim
array-independence without quantifying it; and **no training code or weights** were
released, only the dataset and beamforming scripts.

## 7. Map layout: rectangular vs polar

A rectangular azimuth × elevation image has **two separate defects**. They are worth
keeping apart, because they have different fixes and different costs.

**(a) The azimuth seam.** −180° and +180° are the *same bearing* but sit at opposite
edges of the image. A standard convolution zero-pads there, so a drone crossing due
south is torn across both edges and the network sees a discontinuity that does not
physically exist. **Fix: `padding_mode='circular'` on the azimuth axis.** One argument
to `Conv2d`, no resampling, no data loss. This is cheap and we should just do it — it
does not require the polar remap.

**(b) The pole problem.** As elevation → 90°, all azimuths converge on one physical
direction. The top row spans the full compass but represents nearly a single point, so
pixel distance stops tracking angular distance and a fixed 3 × 3 kernel means something
very different at 4° than at 88° elevation. Convolutions assume a roughly uniform
metric. **Only the polar remap fixes this**: azimuth → angle, elevation → radius,
zenith → centre, i.e. a fisheye view of the sky with a near-uniform metric.

Their reasoning on (b) is sound. But the remap is a trade, not a free win:

| | Rectangular | Polar disc |
|---|---|---|
| Tensor | 91 × 23 = 2093 cells | 46 × 46 = 2116 pixels |
| Usable | all 2093 | 1662 inside the inscribed circle |
| Waste | none | 454 dead corner pixels (21.5%) |
| Resampling | none | 0.79 active pixels per source cell + interpolation |
| Seam | needs circular padding | none |
| Metric near zenith | badly distorted | near-uniform |

So the disc buys a better metric near the zenith at the price of ~21% dead tensor, a
mild loss of average angular resolution, and one interpolation step.

**How much this matters for drones:** less than it sounds. Something 100 m away at 50 m
altitude sits at ~27° elevation, so most frames live in the well-behaved lower band.
High elevations cluster around takeoff and overhead passes, a minority of frames.

**Cost to follow their approach: essentially zero, and it is not a compute question.**
Precompute one fixed (46, 46, 2) sampling grid, then `F.grid_sample` the
(16, 23, 91) feature tensor. Microseconds per frame. Nothing about the download,
feature extraction, or training budget changes.

**Decision: make the map layout a preprocessing flag, not a code fork.** With circular
padding on the rectangular path, rectangular-vs-polar becomes a one-line ablation.
Reporting it as a measured ablation is a *stronger* result than silently adopting
either, because it turns their design claim into something we tested.

## 8. Our own validation run

Script: `utils/validate_dataset.py`. Streamed a 15 s slice of `test_1` via range
requests, rebuilt geometry and GPS alignment from the dataset's metadata, ran
frequency-domain DAS, compared the peak to GPS ground truth.

- Median azimuth error **1.6°**, mean 6.4°, **91% of frames within 10°**
- Median elevation error 5.1°
- Drone distance over the slice: 9–12 m

Caveat: short slice at close range. It proves audio, labels, geometry and coordinate
offsets all line up — it is not a benchmark.

---

## 9. Where the contribution actually lies

Honest ranking, strongest first. None of these is a new architecture — this is a
solid-application project with several genuine novelties, not a new-method paper.

**The headline: can learning replace microphones you didn't buy?** Everything below is
subordinate to this. Section 2 establishes that 6 microphones match the full 1.30 m
aperture, so the other 18 contribute array gain and sidelobe rejection rather than
resolution. That converts a routine "how many mics do you need" ablation into a question
with a mechanism and a deployment payoff:

> A sparse, widely spaced array is cheap and has fine resolution, but its beamformed
> maps are ambiguous — grating lobes create false peaks that classical argmax cannot
> reject. A CNN sees the entire lobe pattern rather than only its maximum, and temporal
> context distinguishes a smoothly drifting drone from flickering false peaks. If that
> recovers most of the accuracy lost by dropping to 6 microphones, then **temporal
> modelling substitutes for hardware**, and an operational sensor costs a quarter as
> much and needs no inter-device synchronization.

Both outcomes are publishable within the project: it works (build cheap arrays and use
temporal context) or it does not (the 18 extra microphones are doing something learning
cannot replicate, which is itself a useful negative). No null-result risk.

This also unifies the proposal — frame stacking stops being a separate bullet and
becomes the *mechanism* for the cheap-array claim.

Ranked contributions:

1. **Aperture vs count, with learning as the compensator** (above, and section 11).
   Operationally actionable, mechanistically motivated, and testable with column
   selection alone.
2. **Temporal context, which they explicitly named as future work.** Their model sees
   one frame at a time and the paper says multi-frame input is the obvious next step. A
   modest change in isolation — which is why it is framed as the tool for (1) rather
   than as an end in itself.
3. **A usable pipeline for a dataset that currently has none.** The authors released the
   data and beamforming scripts but **no training code and no weights**, and their own
   preprocessing produces ~2 TB. Our frequency-domain rewrite (40 MB → 0.13 MB per
   frame, ~300×) plus an open PyTorch implementation is what makes this dataset
   trainable by anyone with a laptop. Reproducibility as a contribution.
4. **An honest detection protocol.** They report detection only as a five-minute
   false-positive proxy measured by standard deviation — no precision/recall, no
   false-alarm rate, no threshold sweep. Their headline claim (67% → 14.9%
   false-positive rate) has never been tested with standard detection metrics. Demoted
   from second place deliberately: disproving someone else's evaluation is a thinner
   story than answering an engineering question, and it reads as adversarial. Keep it as
   a by-product, report it either way.
5. **Testing their polar reparameterization** (section 7) — measuring a design claim
   the original asserts but never ablates.
6. **Speed from the unused GPS velocity labels.** `vx/vy/vz` ship with the dataset and
   nobody has touched them. Novel use of existing labels; also the riskiest item, so it
   stays a secondary output that can fail without harming the main claim.

**Why not lead with "spacing beats count":** that alone is the Rayleigh criterion, and
any reader who knows array processing would shrug. The non-obvious part is the
*substitution* — that a learned temporal prior can stand in for physical aperture
redundancy. Keep the framing on the substitution, not on the resolution scaling.

## 10. Time and compute budget

Estimates. The beamforming figure is measured; the training figures are analytical
(PyTorch is not installed yet, and MPS often underperforms peak on small tensors).
Hardware: MacBook Pro, **M3 Pro, 36 GB** unified memory, PyTorch **MPS** backend.

### One-off costs

| Phase | Estimate | Notes |
|---|---|---|
| Download `test_1` only | ~20 min | 1.1 GB. Do this first, build the pipeline on it |
| Download full audio | **4–8 h, unattended** | ~27 GB at ~1–2 MB/s. Skip the ~19 GB of video. Start day one |
| Feature extraction, full pass | ~1 h | Beamforming measured at 188 frames/s per core; add bandpass filtering and WAV I/O, and budget an hour including the first debugging round |

### Model cost (their configuration: 64 base filters, depth 3, 46 × 46 × 16 input)

| Quantity | Value |
|---|---|
| Parameters | **~7.7 M** (29 MB fp32) |
| FLOPs / forward | ~2.8 G |
| FLOPs / training step | ~8.4 G |
| Training frames | ~34,000 (52 min drone + 4.8 min background) |
| Cost per epoch | ~288 TFLOPs |
| **Est. time per epoch** | **~2.5–10 min** (at 2.0–0.5 TFLOP/s effective) |
| **A 40-epoch run** | **~2–6 h — an overnight job** |

Memory is the thing that makes this comfortable: cached energy maps are **2.1 GB in
fp16** for the training subset (3.2 GB for everything). With 36 GB you load the whole
set into RAM once and never touch disk during training. Data loading is what usually
throttles small models, and we get to skip it.

### The optimization that makes ablations nearly free

Naively, every microphone subset needs its own feature-extraction pass over 27 GB of
WAV, which would blow the budget. Avoid this by **caching per-microphone STFTs instead
of finished energy maps**: 24 mics × 16 bins × complex64 = **150 MB for the entire
dataset**. Steering is the cheap part of beamforming, so any microphone subset, any
direction grid, and the polar remap all become a re-steer over cached STFTs rather than
a re-read of the audio.

Consequence: only **one** pass over the WAVs is ever needed. The microphone ablation,
the grid/layout ablation, and the frequency-bin ablation then cost minutes, not hours.
Do this on day one — it is the single highest-leverage engineering decision here.

Ablations needing no new features at all: temporal stack length (different batching),
rectangular vs polar (grid_sample), frequency bins (subset of cached bins).

### Rough week

Reordered so the headline claim gets tested before the optional extras, not after:

| Day | Work |
|---|---|
| 1 | Kick off full download. Build pipeline on `test_1`, verify labels, cache per-mic STFTs |
| 2 | Full feature pass. Classical brightest-pixel baseline + metrics harness |
| 3 | Single-frame CNN training loop; first real numbers. Falls out for free: their false-alarm claim under proper metrics |
| 4 | Frame stacking (adopt their hyperparameters, don't search). **Headline grid starts:** {6 spread, 6 device-3, 24} × {1 frame, N frames} |
| 5 | Finish headline grid + remaining subsets (device 1, device 4). This is the main result — protect the day |
| 6 | Secondary: polar vs rectangular, speed/Doppler head, analysis and figures |
| 7 | Report and slides |

Overlap the long trainings with writing — they run unattended. The schedule assumes we
do **not** repeat their Optuna hyperparameter search; that is where their compute went,
and we should simply adopt the result (16 bins, 64 filters, depth 3, lr 0.005).

**Feasibility check on the headline grid.** Six configurations at 2–6 h each is 12–36 h,
which does not fit two days at the pessimistic end. Mitigation: comparison runs need a
*shared* budget, not a converged one — run the grid at ~20 epochs with an identical
schedule and seed for every condition, and reserve full-length runs for the two
configurations that end up in the headline claim (24-channel and best 6-channel). Only
relative differences carry the argument, so equal-budget training is the correct
experimental design here as well as the affordable one. State the epoch budget in the
report.

---

## 11. Decisions deliberately deferred (ours, not in the proposal)

### Angular tolerance thresholds

We report median angular error plus hit rate at the dataset's own **10°** label
tolerance, because that is inherited and comparable to the reference. Additional
thresholds if useful in the report:

- **~5°** — about one grid cell, the strict bar near the array's resolution limit.
  Plausible: the paper hits 3.0° on single-source DCASE; our DAS slice hit 1.6° close in.
- **~20°** — the operational bar: enough to swing a camera or point a person, not
  enough for precision aiming.

Reporting several thresholds matters because one number hides whether the model is
precise or merely approximately right. But the exact ladder is presentation, so it
belongs in the final report, not the proposal.

### Microphone-reduction ladder

The full array is four 6-channel recorders, so the meaningful rungs are hardware
breakpoints rather than arbitrary halving:

| Count | What it means physically |
|---|---|
| 24 | Full array, four recorders, needs inter-device sync |
| 12 | Two recorders — half the hardware, still needs sync |
| **6** | **One recorder — removes multi-device synchronization entirely.** The biggest practical cliff, since sync is the fiddliest part of deployment |
| 4 | Tetrahedron — theoretical minimum for unambiguous 3-D direction, and what standard 4-mic SELD arrays (incl. DCASE) use |

**The dominant caveat: count matters less than aperture** — now quantified in section 2.
Since this is the project's headline, the ladder above is the *secondary* axis. The
primary axis holds count at 6 and varies spread, which the recorder grouping gives for
free:

| Condition | Channels | Aperture | Why it is worth a row |
|---|---|---|---|
| Device 1 | 0–5 | 0.31 m | Worst spread, same count — the lower bound |
| Device 4 | 18–23 | 0.49 m | Planar ring: does elevation collapse without vertical spread? |
| Device 3 | 12–17 | 1.09 m | 84% of full aperture, **no sync hardware** — the realistic cheap build |
| Greedy-spread 6 | 2, 9, 11, 15, 16, 17 | 1.30 m | Matches all 24 exactly; the ceiling for 6 mics |
| All 24 | 0–23 | 1.30 m | Same aperture as the row above, 4× the channels — **isolates array gain from resolution** |

The last two rows are the experiment that matters: identical aperture, 4× the
microphones. Any gap between them is *purely* array gain and sidelobe rejection, and
closing that gap with temporal context is the claim. Report which channels were kept for
every number.

Caveat to state honestly in the report: aperture is capped at 1.30 m by the physical
array, so we cannot test *wider*-than-original arrays, only sparser ones. The
extrapolation to "build a cheaper wide array" is an inference, not a measurement.

### Other open choices

- CNN depth, width, normalization, dropout placement — tune after the baseline runs.
- Number of stacked frames for temporal context: start small (3–5) and sweep.
- Frequency bins: paper used 16 over 200–2200 Hz; worth an ablation.
- Rectangular 91 × 23 vs polar 46 × 46 input (see section 7).
- Loss for the sparse positive class: class-weighted cross-entropy first; Dice or
  Tversky as a fallback if imbalance dominates.
- Speed / Doppler head: cheapest thing to cut if time runs short. Baseline to beat is
  simply differentiating the estimated direction track over time.

## 12. Scope cuts already made (keep cut unless time appears)

Dropped from earlier drafts to fit a realistic week: ESC-50 hard-negative injection via
simulated plane waves; SRP-PHAT and MUSIC baselines (brightest-pixel DAS is enough);
cross-array checks on the DDL and UaVirBASE datasets; uncertainty calibration and
Kalman-filter trajectory smoothing.

Known weakness this leaves: one drone model (DJI Air 3), one array, two sites, and all
~10 min of negatives from a single session tail. Detection generalization claims must
stay modest, and the unseen-site test (Test 1) is the honest headline number.
