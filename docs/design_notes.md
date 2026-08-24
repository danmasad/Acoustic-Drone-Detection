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

Channel order in the concatenated array is device order, so **each recorder is already a
6-mic subset** — no rewiring, no re-derivation, just column selection.

Verified across **all four** session geometries (the array was recalibrated between
Oct_11_2024 and the others), because the numbers below are load-bearing for the
proposal:

| Session | Role | Full aperture | Min spacing | Spread-6 aperture | Devices 1/2/3/4 |
|---|---|---|---|---|---|
| Nov_25_2024 f11 | train | 1.30 m | 11.7 cm | 1.30 m | 0.28 / 0.50 / 1.09 / 0.45 m |
| Mar_18_2025 f1 | train | 1.30 m | 11.7 cm | 1.30 m | 0.28 / 0.50 / 1.09 / 0.45 m |
| Mar_18_2025 f2 | test 2 | 1.30 m | 11.7 cm | 1.30 m | 0.28 / 0.50 / 1.09 / 0.45 m |
| Oct_11_2024 f5 | **test 1** | 1.30 m | 17.0 cm | 1.30 m | 0.31 / 0.55 / 1.09 / 0.49 m |

The aperture is **1.30 m in every session**, and a 6-mic subset matches it in every
session, so the proposal's claim holds on both train and unseen-test geometry. Only the
inner radii were recalibrated (`r_` = 0.10, 0.16, 0.23, 0.29, 0.43, 0.63 for the three
newer sessions vs 0.10, 0.17, 0.25, 0.32, 0.42, 0.63 for Oct_11), which shifts min
spacing and the tight-cluster device apertures by 1–4 cm. Nothing structural.

**Discrepancy with the paper, unresolved.** Section 3.2 claims "inter-microphone
distances ranging from 4 cm to 1.1 m". Neither number matches any published session:
measured minimum is 11.7 cm (17.0 cm on test 1) and maximum is 1.30 m. Take the paper's
array spec as approximate and quote our own measurements, which are reproducible from
the shipped JSON.

The table below uses Oct_11_2024 (test 1) since that is where results are reported.

Caveat: the greedy spread-6 set is **not unique** — arms are symmetric, so channels 1
and 2 tie. Fix the set explicitly in code and report it.

"Worst false peak" is the strongest response more than 15° from the true source, relative
to the true peak, across 200–2200 Hz. It is the number that matters, because it is what a
brightest-pixel baseline gets wrong; 0 dB would mean the array cannot tell the true
direction from a phantom at all.

Values below are the **95th percentile over 250 source directions**, from the definitive
dense search in section 14 (`results/subset_search.npz`). An earlier version of this table
quoted a worst-case over only 12 screening directions; those numbers were within a few
tenths of a dB but ranked the middle of the field unreliably, and picked a different
`best-6`. These are the ones to use.

| Subset | Channels | Aperture | Min spacing | Worst false peak | Rows in table above |
|---|---|---|---|---|---|
| All 24 | 0–23 | **1.30 m** | 0.17 m | **−3.6 dB** | all |
| **Best-6 (searched)** | 12, 13, 14, 19, 21, 23 | 0.75 m | 0.43 m | **−3.6 dB** | Z1 + ring |
| Device 3 | 12–17 | **1.09 m** | 0.45 m | **−2.8 dB** | Z1 (r = 0.42, 0.63) |
| Symmetric 6 (top+base) | 0, 1, 2, 15, 16, 17 | 1.30 m | 0.17 m | −1.8 dB | Z3 top + Z1 base |
| 4-mic representative | 12, 13, 14, 22 | 0.75 m | 0.45 m | −1.8 dB | Z1 + ring |
| Device 2 | 6–11 | 0.55 m | 0.18 m | −1.7 dB | Z2 (r = 0.25, 0.32) |
| Device 1 | 0–5 | 0.31 m | 0.17 m | −1.5 dB | Z3 (r = 0.10, 0.17) |
| Greedy-spread 6 | 2, 9, 11, 15, 16, 17 | 1.30 m | 0.55 m | −1.2 dB — **rejected** | mixed |
| Device 4 | 18–23 | 0.49 m | 0.17 m | −0.2 dB, coplanar | Z0 ring |

Three consequences, and they restructured the whole proposal:

1. **Six microphones can match the full array's aperture exactly** (1.30 m). Beamwidth
   scales as λ/aperture, so the greedy-spread 6 has the *same* angular resolution as all
   24 — 30° at 500 Hz, 15° at 1 kHz, 6.9° at 2.2 kHz. The other 18 microphones buy
   **array gain (≈6 dB) and sidelobe/grating-lobe rejection, not resolution.**
2. **The array is already spatially aliased inside the model's own band.** Alias-free
   beamforming needs spacing < λ/2, so 11.7 cm is clean only below **1.47 kHz** while the
   model input runs to 2.2 kHz; greedy-spread 6 at 50 cm is clean only below **343 Hz**.
   Sparse subsets therefore produce genuinely *ambiguous* maps with false peaks, not
   merely blurrier ones. This is exactly the failure a per-pixel CNN with temporal
   context might undo, and argmax cannot.

   Their own paper states this mechanism explicitly (section 4.3, on the 4-mic DCASE
   array): "multiple disconnected detections within the same class can arise due to the
   limited aperture ... and the sidelobe structure of DAS, which can yield competing
   spatial hypotheses within a single frame. This observation motivates temporal context
   (multi-frame input)." **They diagnosed our hypothesis and never tested it** — and
   never on their own 24-channel array, which they only ever ran at full count.
3. **Device 1 vs Device 3 is a free controlled experiment:** same count (6), same
   hardware, same session, aperture 0.31 m vs 1.09 m — a 3.5× difference. Count fixed,
   spread varied. Device 3 alone reaches 84% of the full aperture with **zero
   inter-device synchronization**, which is the cheapest realistic build.

**Correction, measured after the above was written.** Consequence 1 is right that aperture
sets resolution, but it does not follow that we should *choose* a thinned subset by
aperture — and this document then did exactly that. Maximising aperture also maximises the
minimum spacing, which is what causes aliasing, so the two goals fight. Consequence 2
already spotted that greedy-spread is unaliased only below 343 Hz; the subset choice simply
never applied it. Exhaustively searching all 106,183 non-coplanar 6-microphone subsets
gives −3.6 dB, matching all 24, while greedy-spread manages −1.2 dB. Good
subsets are **nested**: short baselines for alias suppression alongside long ones for
resolution. The searched optimum even prefers a 0.75 m aperture to the full 1.30 m.

What survives unchanged: six microphones can match the full aperture; the other eighteen
buy array gain and sidelobe rejection rather than resolution; and section 9's substitution
argument gets *stronger*, because even all 24 leave only 3.6 dB between the true peak and
the worst phantom. Two further measured results: Device 4 is exactly coplanar, so it
resolves azimuth into a stripe spanning every elevation and cannot separate an elevation
from its mirror image; and no 4-microphone subset does better than −1.8 dB, so 4 is a
stress test we expect to fail rather than a deployable option.

## 3. Splits (as the authors defined them)

| Split | Content | Duration |
|---|---|---|
| Train | `training/Nov_25_2024/flight_11` + `training/Mar_18_2025/flight_1` + `test_2/.../flight_2` segment `no_drone_1` | 22 + 30 + ~4.8 min |
| Test 1 | `test_1/Oct_11_2024/flight_5` — **unseen site** | 3 + 1 min |
| Test 2 | `test_2/Mar_18_2025/flight_2` segments `drone` + `no_drone_2` — matched site | 20 + ~4.8 min |

Trap: **all** no-drone audio lives in the tail of one continuous recording
(`flight_2`, seconds 1290–1864), split into `no_drone_1` (train) and `no_drone_2`
(test) so frames are never shared. Do not pre-cut the WAVs; use `skip_seconds` /
`skip_seconds_end` from the JSON as `create_dataset.py` does.

Correction to an earlier note here: there are **no `*_part2.wav` files in the release**. A
directory listing of every session shows only `part1` per recorder, so the 2 GiB WAV cap is a
hard ceiling on how much audio exists, not a subset someone chose. The listing also confirms
the release contains only these four flights — `flight_11` implies earlier flights, but they
were never published.

## 4. Logistics, measured

- License CC BY-SA 4.0. Open HTTP, no registration, HTTP range requests supported.
- 46 GB total; ~27 GB excluding the 360° reference video (~19 GB) which we don't need.
- **Download done.** 28 files, **24.13 GiB**, all byte-verified against
  `utils/dataset_manifest.tsv`; 54 minutes at ~10 MB/s via `utils/download_dataset.sh`
  (2 parallel connections). Much faster than the 1–2 MB/s measured over range requests
  during the feasibility check. Skips the ~19 GB of 360° video.
- `test_1` alone is 1.1 GB — use it to build and debug the pipeline.
- The `*_part1.wav` files are capped at exactly 2 GiB by the WAV format, which is
  1864.1 s of audio; the last segment the paper uses from `flight_2` ends at 1864.0 s, so
  `part1` alone covers everything with 0.1 s to spare.

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

What we actually store is one step further back: per-microphone STFTs (~3.7 GB), not energy
maps, so the maps are re-steered on the GPU at training time. See section 10.

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
Test 1 was held out entirely. Caveat on "16 FFT bins": the paper gives the count and the
band and nothing else, and the code that would fix the rest was never released — how those
16 numbers are formed is our reconstruction (section 16).

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

Hardware: MacBook Pro, **M3 Pro, 36 GB** unified memory, PyTorch 2.13 **MPS** backend.
Now measured rather than analytical.

**MPS gotcha:** `torch.backends.mps.is_available()` returns `False` inside a sandboxed
shell and reports a bogus "macOS 14.0+" error. The GPU is fine; the sandbox blocks it. Run
training unsandboxed.

### One-off costs

| Phase | Cost | Notes |
|---|---|---|
| Download full audio | **54 min, done** | 24.13 GiB at ~10 MB/s, byte-verified |
| Subset beam-pattern search | **35 min, done** | Every non-coplanar 4- and 6-subset on 250 directions; cached in `results/subset_search.npz` |
| STFT cache, full pass | ~1 h | Beamforming measured at 188 frames/s per core; budget an hour including the first debugging round |

### Model cost (their configuration: 64 base filters, depth 3, 16 frequency channels)

Probed with a 6-layer 64-channel conv stack at the real 23 × 91 map size: **1153 frames/s**
(14 ms per step at batch 16), i.e. ~3.2 TFLOP/s effective. That probe is about 3× lighter
per frame than the real model, so scale accordingly.

| Quantity | Value |
|---|---|
| Parameters | ~7.7 M (29 MB fp32) |
| FLOPs / training step | ~8.4 G per sample |
| Training frames | ~34,000 (52 min drone + 4.8 min background) |
| **Measured time per epoch** | **~90 s** |
| **A converged run (cap 60 epochs)** | **~1 h, 90 min worst case** |
| Full experiment grid (37 runs) | ~33 h, unattended, spread over days 3–6 |

This is the change that removed the need for equal-budget training: runs are cheap enough to
train to convergence *and* replicate seeds.

Memory is the thing that makes this comfortable: cached energy maps are **2.1 GB in
fp16** for the training subset (3.2 GB for everything). With 36 GB you load the whole
set into RAM once and never touch disk during training. Data loading is what usually
throttles small models, and we get to skip it.

### The optimization that makes ablations nearly free

Naively, every microphone subset needs its own feature-extraction pass over 24 GB of
WAV, which would blow the budget. Avoid this by **caching per-microphone STFTs instead
of finished energy maps**. Steering is the cheap part of beamforming, so any microphone
subset, any direction grid, and the polar remap all become a re-steer over cached STFTs
rather than a re-read of the audio.

Two refinements to the original sizing, decided when this became code:

- **Cache 381 native bins (200–4000 Hz at the frame's own 10 Hz resolution), ~3.7 GB**, not
  the 16 bins the model consumes (~150 MB). We get one cheap pass over the audio ever, and
  caching only 16 bins would permanently foreclose the frequency-bin ablation and any
  widening of the band. 3.5 GB is a cheap option to buy on a 36 GB machine.
- **Never cache energy maps.** A per-configuration map cache is ~3.4 GB *each*, so five
  subsets would be 17 GB of near-duplicate data. Steering a batch costs ~200 MFLOPs against
  the model's ~8.4 GFLOPs per step — about 2% — so subset, grid, layout and bin count all
  become runtime flags with zero storage. Complex64 matmul works on MPS, so this runs on the
  GPU inside the training loop.

A side benefit worth noting: the cache is 3.7 GB rather than 24 GB, so it is small enough to
copy to a second machine. That makes the experiment grid parallelisable across two people.

Consequence: only **one** pass over the WAVs is ever needed. The microphone ablation,
the grid/layout ablation, and the frequency-bin ablation then cost minutes, not hours.
Do this on day one — it is the single highest-leverage engineering decision here.

Ablations needing no new features at all: temporal stack length (different batching),
rectangular vs polar (grid_sample), frequency bins (subset of cached bins).

### Rough week

Reordered so the headline claim gets tested before the optional extras, not after:

| Day | Work | State |
|---|---|---|
| 0 | Download (54 min) and the full subset beam-pattern search (35 min) | **done** |
| 1 | Geometry and labels ported from the validation script, verified against the known 1.6° median; STFT cache pass | |
| 2 | Classical brightest-pixel baseline + metrics harness | |
| 3 | Single-frame CNN; first real numbers, plus the lr sanity check. Falls out for free: their false-alarm claim under proper metrics | |
| 4 | Frame stacking, then the **order-sensitivity control before the grid**. Headline grid starts: {all 24, device 3, best-6} × {K=1, K=5}, 3 seeds | |
| 5 | Finish the headline grid, then tier 2 (K=3, 9; device 1, 4; symmetric-6; 12-mic rungs). This is the main result — protect the day | |
| 6 | Secondary: polar vs rectangular, frequency bins, speed head, 4-mic floor. Analysis and figures | |
| 7 | Report and slides | |

Overlap the long trainings with writing — they run unattended. The schedule assumes we
do **not** repeat their Optuna hyperparameter search; that is where their compute went,
and we should simply adopt the result (16 bins, 64 filters, depth 3, lr 0.005).

**Superseded: the equal-budget compromise is no longer needed.** The earlier plan was to
run every comparison at a shared ~20-epoch budget, because 2–6 h per run made 12–36 h for
six cells. The measured cost is ~90 min worst case per run, so we train every cell to
convergence instead (early stopping on validation, patience 10, cap 60 epochs) and spend
the headroom on 3 seeds for the six headline cells. Equal-budget training would now be a
self-inflicted confound: a subset that converges more slowly would look worse than it is.
The epoch cap and patience still go in the report.

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

| Condition | Channels | Aperture | Worst false peak | Why it is worth a row |
|---|---|---|---|---|
| Device 3 | 12–17 | 1.09 m | −2.8 dB | **No sync hardware** — the realistic cheap build, and top 1.3% of all subsets |
| **Best-6 (searched)** | 12, 13, 14, 19, 21, 23 | 0.75 m | **−3.6 dB** | The searched ceiling for 6 mics — ties all 24 |
| Device 1 | 0–5 | 0.31 m | −1.5 dB | Worst spread, same count — the lower bound |
| Device 4 | 18–23 | 0.49 m | −0.2 dB | Coplanar: elevation provably collapses. Can the CNN recover it? |
| Greedy-spread 6 | 2, 9, 11, 15, 16, 17 | 1.30 m | −1.2 dB | Maximally aliased — the *hardest* case, not the best one |
| All 24 | 0–23 | 1.30 m | −3.6 dB | The ceiling — **isolates array gain from resolution** |

The comparison that carries the claim is **Device 3 against All 24**: identical sidelobe
structure to within 0.8 dB, a quarter of the channels, and no inter-device synchronization
at all — a build anyone can copy. `Best-6` runs alongside it as the disclosed upper bound:
it ties all 24 exactly, but it spans two recorders (so it is not sync-free) and it was
chosen by searching 106,183 candidates, an advantage all-24 never had. Any residual gap to
all 24 is *purely* array gain, and closing it with temporal context is the claim. Report
which channels were kept for every number.

Note the reversal from an earlier draft: greedy-spread was labelled "the ceiling for 6
mics" and is in fact close to the floor. It keeps a row, but as the maximally ambiguous
stress case — the strongest test of whether a CNN can undo aliasing — never as the headline
cheap array, since a reviewer could trivially pick a better six.

**Measured while building the beamformer, and worth a line in the report.** Device 4 being
coplanar is not just a caveat, it is a free correctness test: a horizontal array must give a
source above its plane and its mirror image below exactly the same response. Measured ratio
for device 4 is **1.00000**, as geometry demands. The interesting part is the other subsets,
where the same test quantifies how weakly *any* thinned array resolves elevation:

| Subset | Flatness | Mirror response / true peak |
|---|---|---|
| all 24 | 1.16 m | 0.19 |
| device 3 | 0.49 m | 0.48 |
| best-6 | 0.43 m | 0.51 |
| device 1 | 0.24 m | 0.54 |
| device 4 | 0.00 m | **1.00** |

So six microphones suppress the mirror image by only ~3 dB where all 24 manage ~7 dB. This
predicts that the six-microphone gap should be larger in elevation than in azimuth, and that
temporal context has more to recover in elevation. Check it in the results, split by axis.

Caveat to state honestly in the report: aperture is capped at 1.30 m by the physical
array, so we cannot test *wider*-than-original arrays, only sparser ones. The
extrapolation to "build a cheaper wide array" is an inference, not a measurement.

### Other open choices

Most of these have since been settled by the build plan; kept here with their resolutions
so the reasoning is not lost:

- CNN depth, width, normalization, dropout — **settled**: adopt theirs (64 base filters,
  depth 3, 3×3, Adam 5e-3), plus circular padding on the azimuth axis so the ±180° seam
  wraps. Then a sanity check rather than a search: three learning rates and one
  weighted-CE vs Tversky comparison at a reduced epoch cap, then freeze for every
  condition. A per-condition search would confound the comparison.
- Stacked frames K — **settled**: K = 5 as the headline against K = 1, with K ∈ {3, 9} as
  single-seed descriptive rows. 5 frames is 0.5 s, roughly the time a drone at 30 m/s and
  100 m needs to move one grid cell, so it is the shortest window in which motion is
  visible at all; 9 frames tests whether more helps or the trail smears.
- Frequency bins — **promoted to a real ablation**, because the 16-bin scheme turned out to
  be our reconstruction rather than something inherited (section 16).
- Rectangular 91 × 23 vs polar 46 × 46 input (see section 7) — kept as a secondary
  ablation, one `grid_sample` away.
- Loss for the sparse positive class: class-weighted cross-entropy first, Tversky as the
  one alternative tested in the step-4 sanity check.
- Speed / Doppler head: still the cheapest thing to cut if time runs short. Baseline to
  beat is simply differentiating the estimated direction track over time.

## 12. Scope cuts already made (keep cut unless time appears)

Dropped from earlier drafts to fit a realistic week: ESC-50 hard-negative injection via
simulated plane waves; SRP-PHAT and MUSIC baselines (brightest-pixel DAS is enough);
cross-array checks on the DDL and UaVirBASE datasets; uncertainty calibration and
Kalman-filter trajectory smoothing.

Known weakness this leaves: one drone model (DJI Air 3), one array, two sites, and all
~10 min of negatives from a single session tail. Detection generalization claims must
stay modest, and the unseen-site test (Test 1) is the honest headline number.

---

## 13. Temporal architecture: why channel stacking, and the control that proves it

**The objection.** Stacking K frames on the channel axis of a `Conv2d` supposedly gives
the model no notion of temporal order — "it can permute the input frames and receive
identical gradients" — so the drift-vs-flicker mechanism would need Conv3D with a causal
kernel, a ConvLSTM/GRU, or causal temporal attention.

**The premise is false.** A `Conv2d` over stacked frames computes

    y = sum_k  W_k * x_k

with a **separate kernel `W_k` per frame slot**. Permuting the `x_k` while holding `W`
fixed changes `y`. Order-invariance would require *tied* weights across k — which is
exactly what Conv3D imposes and channel stacking does not. Channel stacking is the
*less* constrained option in time, not the more constrained one.

Measured with `utils/temporal_stacking_check.py` (reproducible, ~50 s on CPU):

| Check | Result |
|---|---|
| Change in layer output when frames are permuted | **140%** (0% would mean invariant) |
| Change in gradient w.r.t. the input | **81%** |
| Fixed `[+1,-1]` temporal difference kernel, scrambled vs smooth | 1.3× response, untrained |

**And it is learnable, not merely expressible.** Two synthetic tasks, both constructed so
every class has an identical distribution of individual frames — so a single-frame model
is at chance *by construction*, and in Task A both classes share an identical frame
*multiset*, so any order-invariant model is also at chance:

| | Task A: drift vs same frames scrambled | Task B: direction of motion |
|---|---|---|
| 4 frames stacked as channels | **93.8%** | **99.9%** |
| Single frame (their model) | 49.7% | 50.4% |
| Stacked, order shuffled at test | 51.5% | 46.6% |
| Stacked, order **reversed** at test | 93.8% *(correctly unchanged — a reversed smooth track is still smooth)* | **0.1%** |

The 0.1% is the decisive number: reversing the frames makes the model confidently
predict the *opposite* direction. A model blind to order would sit at 50%.

**Why the proposed alternatives are not obviously better here.**

1. *Conv3D ties weights across time*, i.e. assumes translation equivariance along t. But
   our label is the mask for the **current** frame, so "now" and "300 ms ago" play
   different roles and should get different weights. Weight tying is the wrong prior for
   a short window with a privileged endpoint; channel stacking gives slot-specific
   weights for free.
2. *Causality is already free.* The stack is the current frame plus K−1 past frames, so
   nothing from the future can enter. Causal kernels solve a problem we do not have.
3. *Recurrence buys unbounded memory* we do not need over 300–500 ms, at the cost of
   training time and stability on only ~34k frames.

Precedent: DQN stacked four frames as channels precisely so a feed-forward CNN could
infer velocity, and early-fusion video CNNs do the same. This is the standard cheap
approach, not a shortcut invented here.

**A refinement the objection misses, and it matters more.** The dominant cue is probably
*persistence*, not *drift*. Angular motion of the true peak per 100 ms (4° grid cell):

| Range | 5 m/s | 10 m/s | 15 m/s | over a 4-frame stack @10 m/s |
|---|---|---|---|---|
| 20 m | 1.43° | 2.86° | 4.30° | 11.5° = 2.9 cells |
| 50 m | 0.57° | 1.15° | 1.72° | 4.6° = 1.1 cells |
| 90 m | 0.32° | 0.64° | 0.95° | 2.6° = **0.6 cells** |
| 120 m | 0.24° | 0.48° | 0.72° | 1.9° = **0.5 cells** |

Beyond ~70 m the true peak moves **less than one grid cell** over the whole stack, so it
does not visibly drift — it simply *stays put* while sidelobe artefacts come and go.
Persistence is an order-*invariant* statistic, so at the ranges where detection is
hardest the mechanism does not need order at all. Order helps in the near field, where
the target moves several cells per stack. Either way channel stacking covers both cases,
which is the point.

Consequence for wording: "a real drone persists while false peaks flicker" is more
defensible than "drifts smoothly", and true at all ranges.

**The control experiment (run it on day 4, it costs minutes).** Train once with stacked
frames, then at test time (a) shuffle the frame order and (b) reverse it. If accuracy is
unchanged, the model is using only per-frame or order-invariant cues and the temporal
claim is unsupported — report that honestly. If accuracy degrades, quantify by how much;
that number *is* the evidence that temporal structure is doing work. This turns the
objection into a measurement instead of an argument.

**Escalation ladder, only if the control shows stacking is not exploiting time:** add
explicit frame-difference channels (`x_t − x_{t−1}`) as input, which hard-codes the
motion feature; then Conv3D with a causal temporal kernel; then ConvGRU. Each step is a
change to the first block only, so the rest of the pipeline and all array-subset results
stay valid.

---

## 14. How the "best-N" subsets were found

Script: `utils/search_mic_subsets.py`. No audio and no training involved — this is pure
array physics on the geometry from the session's own `alignment_params.json`.

1. **Simulate one source.** Place a unit-amplitude point source at a known direction and
   compute what each microphone receives: a pure phase shift, `exp(-2πif·τ_m)`, where τ_m
   is the extra travel time to microphone m. No noise, no drone spectrum — we are measuring
   the array, not the task.
2. **Beamform the whole grid.** Steer to all 2093 directions and sum energy over 16
   frequencies spanning 200–2200 Hz. The result is the array's point spread function: one
   true peak plus every artefact the geometry produces.
3. **Score it.** Take the strongest response more than 15° from the true direction, divided
   by the true peak, in dB. That is the margin a brightest-pixel baseline has to work with.
   Repeat for 12 source directions (4 azimuths × elevations 10°, 30°, 60°) and keep the
   **worst** case, so a subset cannot win by being good in one direction only.
4. **Reject flat subsets.** Compute the smallest singular value of the centred positions —
   the spread perpendicular to the best-fit plane. Below 0.15 m the microphones are
   effectively coplanar and elevation is unrecoverable in principle, so those subsets are
   excluded rather than scored. This removed most 4-microphone candidates, including a
   hand-picked one that scored −0.2 dB.
5. **Enumerate everything.** C(24,4) = 10,626 and C(24,6) = 134,596, of which 2,142 and
   106,183 survive the flatness filter. Exhaustive, so "best" means best, not best-found.

The trick that makes exhaustive search cheap: the beamformer output for a subset is just a
**masked sum** of one precomputed tensor, since each microphone contributes an independent
term. So all subsets of a size are evaluated as a single matrix product against a binary
mask matrix, in chunks. Whole search: 108 s on CPU.

### Does the winner generalize? Held-out validation

Picking the best of 106,183 candidates using 12 directions invites selection overfitting:
the top scorer could simply have been lucky on those 12. Stage 2 therefore re-scores the
top 200 on **200 directions drawn uniformly and disjoint from the screening set**.

Note first that *distance never enters*. The beamformer assumes plane waves, so only
bearing matters — a drone at 40 m and at 400 m on the same bearing give identical geometry.
The generalization question is over direction, not position.

| Subset | Screen (12 dirs) | Held out, worst | Held out, median |
|---|---|---|---|
| all 24 | −3.8 dB | −3.3 dB | −3.9 dB |
| best-6 `12,13,14,18,20,22` | −3.6 dB | **−3.5 dB** | −3.9 dB |
| device 3 | −2.9 dB | −2.5 dB | −3.0 dB |
| greedy-spread 6 | −1.0 dB | −1.0 dB | −1.6 dB |
| 4-mic `12,13,14,22` | −1.9 dB | −1.7 dB | −1.9 dB |

**At six microphones the search generalizes.** The screen winner holds its value (−3.6 to
−3.5 dB) and ranks 2nd of 200 on unseen directions; the held-out winner is its own mirror
image under the array's 3-fold symmetry. Even the *worst* of the top 200 scores −2.3 dB on
held-out directions, so the screen located a good region rather than a lucky point. And the
headline result survives the harder test: **best-6 matches all 24 within 0.2 dB on the worst
case and exactly on the median**, across 200 directions.

**At four microphones the search is meaningless.** Rank correlation between screen and
held-out scores is **+0.00**, and the screen winner falls to rank 157 of 200. The whole
shortlist spans only −1.8 to −1.4 dB, so the differences the screen was ranking are
direction-to-direction noise, not real structure. Consequence for reporting: there is no
"best 4" and we must not claim one. Call it a *representative* 4-microphone subset, note
that all 4-mic subsets are equivalent within noise, and treat the row as the floor.

### Single-stage dense search: the definitive answer

`utils/search_best_dense.py` scores **every** non-coplanar subset on **250 directions**
(disjoint from both earlier sets), ranking by the **95th percentile** of the false-peak
level rather than the maximum, since a max over 250 samples is dominated by one unlucky
direction. Runtime 35 min; all scores are stored in `results/subset_search.npz` so plots
never recompute. Figure: `figures/subset_distribution.png`.

| Subset | p95 | max | median | Position among 106,183 |
|---|---|---|---|---|
| best-6 `12,13,14,19,21,23` | **−3.6 dB** | −3.5 | −3.8 | the optimum |
| **all 24** | **−3.6 dB** | −3.3 | −3.9 | — |
| device 3 | −2.8 dB | −2.6 | −3.0 | **top 1.3%** |
| symmetric-12 (12 mics) | −2.8 dB | −2.6 | −3.1 | — |
| device 1+3 (12 mics) | −2.4 dB | −2.1 | −2.9 | — |
| symmetric-6 | −1.8 dB | −1.4 | −2.3 | top 33% |
| device 2 | −1.7 dB | −1.4 | −2.6 | top 44% |
| device 1 | −1.5 dB | −1.5 | −1.6 | median |
| greedy-spread 6 | −1.2 dB | −0.9 | −1.5 | bottom 21% |
| device 4 | −0.2 dB | −0.1 | −2.2 | **worst of all 106,183** |

Population: best −3.6, median −1.6, worst −0.6 dB — a 2.9 dB spread, so subset choice
matters roughly as much as the difference between 6 and 24 microphones.

Four results worth carrying into the report:

1. **Six microphones match twenty-four exactly.** Both sit at −3.6 dB p95. The cheap-array
   premise is not approximately true, it is exactly true on this metric.
2. **The cheap two-stage search had already found the true optimum.** The dense winner is
   the mirror image of the two-stage winner under the array's 3-fold symmetry, and both
   score −3.6. So the screening shortcut was sound at n=6 — which we only know because we
   checked.
3. **Device 3 is in the top 1.3%.** The deployable single-recorder choice is not merely
   reasonable, it beats 98.7% of all possible six-microphone arrangements.
4. **More microphones can be worse.** `device 1+3` uses 12 microphones and scores −2.4 dB,
   *worse* than `device 3`'s 6 at −2.8 dB. Adding device 1's tight inner cluster with
   uniform weighting raises the sidelobe floor relative to the main lobe. "More channels is
   better" is false for unweighted delay-and-sum, and it means the 12-microphone rung may
   produce a negative result — which is worth reporting, not hiding.

At four microphones the population spans only 0.8 dB (best −1.8, median −1.7), confirming
from the full distribution what the held-out test suggested: **there is nothing to choose
between 4-microphone arrangements.**

### Why the searched optimum is a bound, not the headline

Three reasons the deployable subset leads and `best-6` follows:

1. **`best-6` is not sync-free.** Channels 12, 13, 14 belong to device 3 and 18, 20, 22 to
   device 4, so it spans **two recorders** and still needs inter-device synchronization —
   the very cost the cheap-array story is about removing. Excellent beam pattern, mediocre
   deployment value.
2. **Selection asymmetry.** We chose the best of 106,183 six-microphone arrangements, but
   there is exactly one 24-microphone arrangement and nobody optimised it. "best-6 matches
   all 24" therefore compares an optimised six against an unoptimised twenty-four. The
   parity is real but flattered, and the honest fix is to report the whole distribution
   (`utils/plot_subset_distribution.py`) rather than only the maximum.
3. **It is not a buildable prescription.** A searched subset of this particular tripod tells
   nobody how to build a cheaper sensor. "One recorder's worth of microphones spread over
   about a metre" does.

The clean statement that survives all three: within a single recorder there are only four
possible choices, and **device 3 is the best of them** (−2.5 dB against −1.5, −1.4 and −0.1
on held-out directions). So device 3 is not a hand-wave — it is *the* optimal sync-free
six-microphone array on this hardware, and it lands within about a decibel of both the
global best six and the full twenty-four.

Keep the search anyway. It already earned its keep by falsifying the greedy-spread choice,
which would otherwise have been the headline cheap array and a straw man, and keeping the
optimum in the table forecloses the obvious attack that we simply picked a bad six.

**Generalization gaps still untested.** The source here has a flat spectrum, no noise, and
one source at a time. A real drone has harmonic structure concentrated at blade-pass
frequencies, and wind noise is concentrated low. Once the cache exists (day 2), reweight the
frequency average by the measured drone spectrum and re-check the choice — it is a minutes-
long rerun. And the deeper caveat stands: a subset optimal for classical DAS is not
necessarily optimal for a CNN, which may prefer structured, predictable artefacts over
merely smaller ones. That is why the claim rests on **device 3**, chosen on hardware grounds
and immune to all of this, with best-6 reported as the searched upper bound.

**Caveat on using the result.** Picking the subset that optimises a classical DAS criterion
is itself a form of search, on a quantity correlated with the task. That is why the grid
carries *two* six-microphone rows: `device 3`, which is a physically deployable single
recorder chosen for hardware reasons, and `best-6`, which is the searched optimum reported
as an upper bound. The claim should rest on the deployable one.

Also: this analysis is noise-free and single-source. It captures *ambiguity*, not
*sensitivity* — all 24 microphones still collect roughly 6 dB more signal-to-noise from
averaging, which is what matters for detecting a quiet drone at long range. A subset can
match all 24 on sidelobe structure and still lose on detection range.

## 15. Experiment grid

Subsets, with the worst-case false peak from section 14:

Sidelobe is the 95th percentile over 250 directions from the dense search (section 14) —
the definitive number.

| Name | Channels | Sidelobe | Aperture | Character |
|---|---|---|---|---|
| best-6 | 12, 13, 14, 19, 21, 23 | **−3.6 dB** | 0.75 m | global optimum; **ties all 24** |
| all 24 | 0–23 | −3.6 dB | 1.30 m | full hardware, needs 4-device sync |
| device 3 | 12–17 | −2.8 dB | 1.09 m | one recorder, no sync — **top 1.3%** |
| symmetric-12 | 0–2, 6–8, 12–14, 15–17 | −2.8 dB | 1.30 m | four rings, 12 mics |
| device 1+3 | 0–5, 12–17 | −2.4 dB | 1.30 m | 12 mics yet *worse* than device 3 |
| symmetric-6 | 0, 1, 2, 15, 16, 17 | −1.8 dB | 1.30 m | structured, full aperture |
| device 1 | 0–5 | −1.5 dB | 0.31 m | same count, worst spread |
| greedy-spread 6 | 2, 9, 11, 15, 16, 17 | −1.2 dB | 1.30 m | maximally aliased stress case |
| device 4 | 18–23 | −0.2 dB | 0.49 m | coplanar; worst of all 106,183 |
| 4-mic (representative) | 12, 13, 14, 22 | −1.8 dB | 0.75 m | floor; no "best 4" exists |

**Tier 1 — headline, 3 seeds each.** This is where the claim lives, so it gets error bars.

| Subset | K | Runs | Question it answers |
|---|---|---|---|
| all 24 | 1 | 3 | reproduces their published single-frame model |
| all 24 | 5 | 3 | does context help even with full hardware? |
| **device 3** | 1 | 3 | the deployable single-recorder build, no context |
| **device 3** | 5 | 3 | **does context buy back the hardware gap?** |
| best-6 | 1 | 3 | searched upper bound, no context |
| best-6 | 5 | 3 | searched upper bound, with context |

Device 3 carries the claim; best-6 is the disclosed upper bound. Order matters here because
best-6 spans two recorders and so is not sync-free (section 14) — it answers "how good can
six get", while device 3 answers "what can you actually build".

**Tier 2 — descriptive, 1 seed each.** These describe rather than claim.

| Subset | K | Runs | Purpose |
|---|---|---|---|
| all 24, best-6 | 3, 9 | 4 | saturation curve; K=9 is the 900 ms latency limit |
| device 1 | 5 | 1 | count held at 6, spread at its worst |
| device 4 | 5 | 1 | can the CNN recover elevation a coplanar array cannot? |
| symmetric-6 | 5 | 1 | structured alternative at full aperture |
| device 1+3, symmetric-12 | 5 | 2 | the two-recorder rung |

**Tier 3 — controls and secondary.**

| Item | Runs | Note |
|---|---|---|
| Order control: shuffle and reverse at test | 0 | evaluation only, on the first device-3 K=5 checkpoint, **before** the grid runs |
| Learning-rate sanity check | 3 | 25-epoch cap, ~25 min each |
| Weighted-CE vs Tversky | 1 | 25-epoch cap |
| Polar disc vs rectangular | 1 | best-6 at K=5 |
| Frequency-bin count: 4, 8, 32 against the default 16 | 3 | now our choice, not inherited (section 16), so it earns a real sweep. Run on device 3 at K=5, one seed, everything else frozen |
| Speed head from GPS velocity | 1 | stretch goal |
| best-4 floor | 2 | K=1 and K=5, run last |
| greedy-spread stress row | 1 | optional: hardest case for the CNN |

**39 runs, roughly 35 h of unattended compute** at ~1 h per converged run. Cut order if
that estimate proves optimistic: tier 3 secondary first, then tier 2, never the eighteen
headline runs.

Protocol for every run: model selection on **test 2 only**, test 1 sealed until the numbers
are final and then evaluated once for the headline rows. Early stopping on test-2 AUPRC,
patience 10, min improvement 0.001, floor of 15 epochs, hard cap 60, restore the best
checkpoint, constant learning rate — identical everywhere, because that identity is what
makes the rows comparable. Metrics per run: precision–recall and AUPRC, false alarms per
hour, median angular error, hit rate at 5/10/20°, and error binned by distance.

---

## 16. The 16 frequency bins are our reconstruction, not their spec

Worth recording because section 6 lists "16 bins" among the hyperparameters we adopt from
the paper, which overstates what the release actually pins down.

The paper describes the model input as `2E × 2E × F` with `F = 16` over 200–2200 Hz, and
says nothing about how those 16 numbers are formed. It is not recoverable from the code
either: `create_dataset.py` contains no STFT, bin selection or frequency logic at all — it
bandpass-filters and beamforms to full waveforms, which is exactly why its output reaches
~2 TB — and the `hyperparams.yaml` that the README references was never shipped. The
training code is the part they withheld.

Our reading: **16 equal-width 125 Hz sub-bands, each averaged from a group of about 12–13
native 10 Hz bins.** Averaging rather than picking 16 sparse bins, because averaging keeps
all the energy and is far less sensitive to exactly where a bin boundary falls. Document it
as a reconstruction in the report, not as their setting.

Two consequences: the frequency-bin count is promoted from a nice-to-have to a real
ablation, since it is a design choice we are making; and the STFT cache holds all 381 native
bins so the binning stays a runtime flag rather than something baked into 24 GB of
preprocessing.

---

## 17. First real numbers: the classical baseline, and what it does to the claim

Brightest-pixel delay-and-sum on test 2 (14,797 frames: 11,927 with a drone, 2,870 without),
detection scored on the peak's absolute energy. `results/baseline.json` has the full output.

| Subset | Median angular error | Azimuth | Elevation | Within 10° | AUPRC | 0–30 m | 30–60 m | 60–90 m |
|---|---|---|---|---|---|---|---|---|
| all 24 | 37.2° | 27.0° | 11.1° | 33% | 0.883 | 9.9° | 56° | 86° |
| **device 3** | **22.1°** | 23.7° | 5.5° | 38% | 0.898 | 9° | 35° | 77° |
| best-6 | 35.1° | 25.7° | 11.2° | 26% | 0.890 | 14° | 44° | 68° |

Two things fall out, and the second one matters.

**The predicted range failure is confirmed, hard.** Error grows 8.6× from the near field to
60–90 m. That is the paper's headline weakness reproduced with our own code and our own
metrics, and it is what a learned model has room to fix.

**Detection by brightest-pixel energy is worthless, as it should be.** AUPRC 0.883 against a
positive-class prior of 0.806, and the best-F1 operating point simply labels every frame a
drone — 36,000 false alarms per hour, i.e. all of them. This is the paper's 67% false-positive
figure restated under honest metrics. It also means the CNN's detection bar is *the prior*,
not their number, and we should say so.

**The awkward one: at the classical level, six microphones already beat twenty-four.**
Device 3 halves all-24's median error. That was not predicted, and it is worth understanding
before the grid runs: all 24 has the narrowest main lobe, but it also includes device 4's flat
coplanar ring and the tight inner clusters, and its 0.17 m minimum spacing means it is
aliased above 1 kHz across most of the model band. Uniform, unweighted delay-and-sum gives
every microphone equal say, so bad geometry actively hurts. The same effect explains why
`device 1+3` scores worse than `device 3` alone on sidelobes despite twice the microphones.

Implication for the framing: **do not promise that a 6-vs-24 gap exists and gets closed.**
The honest question is what the *learned* model does with each array, and the answer may be
that the cheap array was never behind to begin with — which is a stronger operational result
than the one we set out to prove, not a weaker one. The claim should be stated as "context
plus a cheap array matches full hardware", with the classical numbers shown alongside so the
reader sees that uniform DAS was already leaving the extra 18 microphones' value on the
table.

---

## 18. First CNN runs: localization works, detection is data-limited

Three short runs of device 3 at K = 1 (their configuration: 64 base filters, depth 3, Adam
5e-3, weighted cross-entropy, 16 sub-bands). Not converged — 2–3 epochs — so read the
direction of travel, not the values.

| | DAS baseline (device 3) | CNN, 2 epochs |
|---|---|---|
| Median angular error | 22.1° | **6.2°** |
| Within 10° | 38% | **64%** |
| 30–60 m | 35° | 4° |
| 60–90 m | 77° | 5° |
| Detection AUROC | 0.689 | 0.669 |
| Negatives rejected at 95% recall | 10% | 2% |

**Localization: the gate is passed by a wide margin**, and the range dependence is gone —
error is flat from 30 to 90 m where DAS degrades monotonically. That is the paper's central
weakness, fixed, on a six-microphone array.

**Detection is not learning anything generalizable, and the cause is the dataset, not the
model.** All the negative audio in the release is one continuous ambient tail: 4.8 minutes for
training, the adjacent 4.8 minutes for test. 92% of training frames contain a drone. Balanced
class weighting and a max-pooled presence head moved AUROC to 0.86 after one epoch and then it
collapsed to 0.67, which is the signature of memorizing 2,869 negative frames. Section 12
already flagged this as the known weakness; it is now measured.

Options, in increasing cost: report detection honestly as data-limited and keep localization
as the headline; or reinstate a cut-scope item and manufacture hard negatives (ESC-50 clips
injected as plane waves, or spectrally scrambled ambient), which buys a real false-alarm
number at the price of a day and an unverifiable realism assumption.

### Mined negatives, and the hard limit they do not remove

Every session's recorders were started before the flight, so each WAV opens with real
no-drone audio the dataset's splits never used. Verified ambient rather than drone by
beamformed peak contrast (`utils/probe_pretakeoff.py`): every pre-takeoff window sits inside
the range of the labelled ambient tail and at or below its own session's drone audio. A
20 s margin before takeoff is discarded for motor spin-up and handling noise, and each
window inherits the split of its own session, so no test audio can leak into training.

| Split | Was | Now | Scenes |
|---|---|---|---|
| train | 4.8 min | **8.2 min** | 1 → **3** |
| test 2 | 4.8 min | **5.6 min** | 1 → 2 |
| test 1 | **none** | **1.0 min** | 0 → 1 |

The diversity change matters more than the minutes: training negatives now come from three
separate recordings on three days instead of one continuous tail, which is what the collapse
from AUROC 0.86 to 0.67 was diagnosing. Test 1 gains negatives for the first time, so an
unseen-site false-alarm number becomes possible at all.

**What it does not fix.** With 100 ms frames, one false positive in `T` of ambient audio is
`1/T` false alarms per hour, so the finest rate we can resolve is **10.7/h on test 2** and
**58/h on test 1** — 3.5/h even pooling every minute of ambient audio in the release. Any
operationally interesting target (a handful per hour) is below our measurement floor. This is
not a modelling problem and no amount of tuning touches it.

Consequences for what we claim: report detection as AUROC and precision-recall, state the
resolution floor next to any false-alarms-per-hour figure, and keep **localization as the
headline**, where 11,927 drone frames make the comparison well powered. Still worth saying
plainly in the report that this is the first real precision-recall evaluation on this dataset
— the reference paper's detection claim rests on five minutes of one recording, and ours rests
on seventeen across four.

If a genuine false-alarm battery is wanted later, ESC-50 is the right source (2,000 clips,
50 environmental classes, ~2.8 h) injected as plane waves, reported separately as *confuser
rejection* rather than as a field false-alarm rate. Al-Emadi is the weaker option for this:
mono 16 kHz indoor clips of other drone models, and its negatives are indoor ambience.

**Measured compute, replacing the estimates in section 10.** Batch size is irrelevant on MPS
here — the model is compute-bound — but width matters a lot:

| Configuration | Throughput | Epoch | Parameters |
|---|---|---|---|
| 64 base filters (theirs) | 219 frames/s | 156 s | 7.9 M |
| 32 base filters | 512 frames/s | 68 s | 2.0 M |

At 64 filters a converged run is 60–90 min and the full 39-run grid is 40–50 h, over the 35 h
budget. At 32 filters the grid is ~20 h, and since the 64-filter model overfits visibly
(training loss falling while validation loss rises), the narrower model may also generalize
better. Their width was tuned by Optuna for their own single-frame setup, which is a reason to
respect it but not to treat it as sacred.

## 19. Detection as a per-channel question: the split, and what it bought

The map-based presence head reads a beamformed energy map, so anything injected as a plane wave
carries a spatial signature our rendering invented. Rendered clips have a perfectly coherent
wavefront and no per-microphone noise, and identical duplication across channels — a signal no
physical source produces — would be worse still: the model would learn *coherent means
synthetic means no drone* and generalize nowhere. Whichever direction the rendering differs from
reality, the difference is a shortcut.

A detector that reads **one microphone at a time** has no cross-channel structure to exploit, so
mono audio needs no rendering at all and the confound disappears. Three further consequences
argued for trying it: 24 examples per frame instead of one, detection made independent of
microphone count (which isolates the 6-versus-24 comparison to localization, where the physics
actually differs), and a model small enough to be free. Implemented in `src/detector.py` as a
four-layer 1-D CNN over the full 381-bin log-magnitude spectrum, 86k parameters, frame decisions
by averaging per-channel logits.

| Detector | Parameters | Examples/epoch | test 2 AUROC | Reject @ 95% recall |
|---|---|---|---|---|
| Map-based presence head | 7.9 M | 36,127 | 0.669 | 1.6% |
| Per-channel spectral | **86 k** | **867,048** | 0.662 | **6.6%** |

**It did not help, and the way it failed is the useful part.** Ninety-two times fewer parameters
and twenty-four times more examples land on the same AUROC, and the channel sweep shows why: one
microphone scores 0.681, twenty-four averaged score 0.662. Averaging over channels does not
help, so the per-channel errors are almost perfectly correlated — the 24 channels of a frame are
near-duplicates of one acoustic event, and the effective sample size is the number of *scenes*,
not the number of channel-frames. Training loss falls to 0.022 while validation AUROC stalls
near 0.66: the model is memorizing the ambient signature of the three training recordings.

So detection is bounded by scene diversity, and no architecture reaches past that bound. This
confirms section 18's diagnosis from the opposite direction and settles the split's real purpose:
the per-channel detector is not a better model, it is the **only clean way to let outside mono
audio in**. ESC-50 mixed into our own ambient channels — real noise floor, added confuser
energy, no invented geometry — is now a legitimate experiment rather than a shortcut generator.
Small consolation prize meanwhile: at 95% recall it rejects 6.6% of negatives against the map
head's 1.6%, four times better, for 1% of the parameters.

Reporting both is the honest form, and cheap: the map model is trained for localization anyway,
and the detector adds ~3 min per run.

### ESC-50 as confusers, built but not yet trained on

`src/esc50.py` mixes ESC-50 into our own spectra rather than substituting for them. Three
choices in it are worth remembering:

- **Mixed, never used alone.** A clip on its own carries its own gain, microphone and room, so
  "no drone" would be predictable from provenance. Added on top of a real frame, our array's
  noise floor sits under every example and the confuser only adds energy.
- **Mixed into positives too**, on a fraction of frames. Otherwise "contains an added sound"
  becomes a perfect cue for the negative class and we have rebuilt the shortcut somewhere new.
- **Spectral-domain mixing.** Our frames are non-overlapping 4800-sample rectangular windows,
  so the per-frame STFT is linear and adding spectra *is* adding waveforms. Confusers can be
  injected during training without re-reading 24 GB of audio. Level is set per frame as a
  confuser-to-frame energy ratio drawn from −5 to +20 dB, so it adapts to loud and quiet frames.

Fold 5 is held out for the reported number, folds 1–4 feed training mixes. The eight classes
worth reporting separately are helicopter, airplane, chainsaw, engine, vacuum cleaner, washing
machine, hand saw and wind — rotating machinery and harmonic stacks in our band. A mean over
fifty classes would hide exactly the cases that matter.

This is for the per-channel detector only. The same mixture fed to the map-based model would
carry an invented wavefront, which is the shortcut this whole section exists to avoid.

## 20. Step 5a: the stack uses the past, and the payoff is undecided

`device3`, `base=32`, seed 0, K=1 against K=5, with three test-time interventions on one set of
trained K=5 weights. Reproduce with `utils/check_order.py`.

| Condition | Within 10° | Median | Median <60 m | Detection AUROC |
|---|---|---|---|---|
| K=1 baseline | 71.5% | 5.60° | 5.61° | 0.733 |
| **K=5 causal** | **72.1%** | 5.77° | 6.06° | **0.860** |
| K=5 freeze (current frame ×5) | 67.4% | 6.36° | 6.26° | 0.865 |
| K=5 shuffle | 71.9% | 5.78° | 6.09° | 0.860 |
| K=5 reverse | 71.9% | 5.86° | 6.15° | 0.860 |

**The mechanism is real.** Replacing the five frames with five copies of the current frame costs
4.7 points of localization. The model genuinely looks back; the extra input channels are not
just capacity.

**Order does not matter, history does.** Shuffling and reversing cost 0.2 points, near zero, and
near-field rows move by 0.09°. Exactly what section 13 predicted: beyond ~70 m the true peak
moves under one grid cell per stack, so the cue is *persistence*, which is order-invariant by
construction. Wording follows: a drone **persists** while false peaks flicker. Never "drifts
smoothly", and the model's own behaviour now says so.

**The payoff is undecided, and that is the honest state.** K=5 beats K=1 by 0.6 points, well
inside single-seed noise — epoch-to-epoch swings on this metric are several points. So the stack
uses history, and that use does not yet buy accuracy. The 3-seed headline grid is the
measurement that decides, and a null there is a real answer to the question the reference paper
left open rather than a failure.

**One thing that is not what it looks like.** Detection AUROC jumps from 0.733 to 0.860 with
K=5, and negative rejection at 95% recall goes from 12% to 33%. Tempting to credit temporal
context — but freezing the stack leaves detection at 0.865, unchanged. Whatever produces that
gain, it is not history. Treat it as run-to-run variance until three seeds say otherwise.

**Three bugs found here, and two of them wore the same disguise.** An AUROC of exactly 0.5000
means every score is tied, and we hit that twice for unrelated reasons:

1. *Saturation.* Scoring used `sigmoid(logit)`, which reaches exactly 1.0 in float32 above a
   logit of about 17, so confident predictions collapsed into one value. Metrics only need the
   ranking, so scores are now raw logits.
2. *A dead head.* At lr 5e-3 the presence head's ReLU layer died outright in the `all24` seed-0
   run: all 128 units sat at negative pre-activations for every input (mean −7.6, max −0.50),
   the logit was frozen at the output bias, and AUROC read 0.500 from epoch 0 through 17 while
   localization trained normally at 72%. A dead ReLU passes zero gradient and a single hidden
   layer offers no way back, so it never recovers. Now `LeakyReLU(0.01)`, and the epoch log
   prints the presence-logit spread so a collapse is visible in the first minute rather than in
   a finished run. Seed 1 of the same cell was healthy, which is what makes this worth writing
   down: it is a seed-dependent failure that a single run would never have revealed.
3. *Width.* `base=32` localizes *better* than the paper's 64 (72% against 63% within 10°) as
   well as 2.3× faster, so the narrower model is what the grid uses. The width they tuned by
   Optuna was tuned for their single-frame setup.

The three runs trained before the dead-head fix are kept under `runs/_relu_head/` rather than
deleted, and the grid was restarted so all 18 cells share one architecture. Comparability across
cells is the whole point of the protocol; a mixed-architecture grid would measure nothing.

## 21. The headline grid: what 18 runs actually say

Model selection on test 2, test 1 still sealed, `base=32`, 3 seeds per cell, 808 min of compute,
no failed cells. Reproduce the tables with `utils/summarize_grid.py`.

**Frames within 10°, mean ± sample sd over 3 seeds:**

| Array | Microphones | Recorders | Sidelobe score | K=1 | K=5 | K=5 − K=1 |
|---|---|---|---|---|---|---|
| all24 | 24 | 4 | −3.6 dB | 70.5% ± 1.5 | 70.1% ± 0.9 | −0.4 |
| **device3** | **6** | **1** | −2.8 dB | **69.3% ± 0.7** | **71.1% ± 1.8** | **+1.8** |
| best6 | 6 | 2 | −3.6 dB | 52.1% ± 0.8 | 52.8% ± 1.2 | +0.6 |

Pooled within-cell seed spread is ±1.2 pp, and that is the yardstick for everything below.

### Result 1: six microphones on one recorder match twenty-four

The device3-minus-all24 gap is −1.2 pp at K=1 and +1.0 pp at K=5, both inside seed noise
(±1.7 and ±2.1 pp). A single 6-microphone recorder with no inter-device synchronization equals
the full array at four times the hardware. This is the operational claim the project was for,
and it is now measured three times per cell rather than argued.

Note what it does to the original framing. We expected to need temporal context to close a gap;
there is no gap to close. The cheap array was already level at K=1, which is a stronger result
about hardware and a weaker one about temporal context.

### Result 2: temporal context does not reliably help

+1.8 pp for device3 against a ±1.2 pp seed spread, −0.4 pp for all24, +0.6 pp for best6. The
difference of differences is +2.2 pp in device3's favour, so if context helps anywhere it is on
the cheap array, which is the direction the mechanism predicts. But one cell at barely over the
noise floor is not a finding, and median error moves by 0.0°.

This is a direct answer to the reference paper's own suggested future work: on this dataset,
stacking frames does not measurably improve direction finding. Stated as a null result with the
seed spread beside it, which is more than their five-minute single-recording claim carried.

### Result 3: the searched-optimal array loses by 18 points, and that is the interesting one

`best6` scores −3.6 dB on worst-case sidelobes, tied with all 24 microphones and the best of all
106,183 non-coplanar 6-subsets. It lands at 52% against device3's 69%. Eighteen points, fourteen
times the seed spread, in the opposite direction to what the beampattern predicts.

The first explanation we reached for was clock sharing: `best6` is 3 microphones from device 3
plus 3 from device 4, while `device3` is one recorder, and cross-device phase rests on a single
hand-set integer sample correction. Section 22 tested that and **refuted it**. The real
explanation is selection on a proxy, and it is a better story.

### Result 4: order does not matter, history does — replicated

Averaged over all nine K=5 runs, freezing the stack to the current frame costs 4.3 pp, while
shuffling costs 0.1 and reversing 0.2. The step 5a finding holds across three arrays and three
seeds, so the wording is settled: **a drone persists while false peaks flicker**, and the model's
own behaviour says the direction of time is irrelevant to it.

### Result 5: map-based detection is not usable, and the per-channel detector wins by default

Detection AUROC over seeds: all24 0.727 ± 0.135, device3 0.586 ± 0.220, best6 0.671 ± 0.074.
One device3 K=5 run scored **0.288**, meaning its presence head ranked no-drone frames *above*
drone frames on the unseen scene. Rejection at 95% recall swings ±7–9 pp between seeds of the
same cell. With 4.9 min of training negatives from three scenes, the head memorizes ambient and
inverts on a new one; seed variance dwarfs every effect we could report.

So detection is reported from `src/detector.py` (per-channel, AUROC 0.662, stable), the map
head's presence numbers are reported only as the paper-faithful comparison with their spread
shown, and the scene-diversity bound from section 18 is quoted next to both.

## 22. Why the searched-optimal array failed: selection on a proxy, not clock sharing

Five more K=1 runs, 176 min, chosen so that the two candidate explanations make opposite
predictions. Reproduce with `utils/summarize_grid.py`.

| Array | Mics | Recorders | Largest group on one clock | Sidelobe | Within 10° |
|---|---|---|---|---|---|
| device1_3 | 12 | 2 | 6 | −2.4 dB | **73.1%** |
| all24 | 24 | 4 | 6 | −3.6 dB | 70.5% |
| device3 | 6 | 1 | 6 | −2.8 dB | 69.3% |
| symmetric6 | 6 | 2 | 3 | −1.8 dB | 64.0% |
| device1 | 6 | 1 | 6 | −1.5 dB | 59.0% |
| device4 | 6 | 1 | 6 | −0.2 dB | 54.5% |
| quad4 | 4 | 2 | 3 | −1.8 dB | 54.2% |
| **best6** | 6 | 2 | 3 | **−3.6 dB** | **52.1%** |

**The clock-sharing hypothesis is dead.** `symmetric6` spans two recorders and beats
single-recorder `device1` by 5 points and `device4` by 9.5. `device1_3` spans two recorders and
is the best array we have measured. Crossing recorders costs nothing here — the dataset's
integer sample corrections are good enough — and the prediction we wrote down before running was
simply wrong. Recording it as refuted rather than quietly dropping it.

**What is actually going on.** Beampattern score predicts accuracy well: correlation −0.80
across every array in the table *except* the one that was chosen by maximizing it. Include
`best6` and the correlation collapses to −0.36. The proxy is not broken; it is broken *for its
own winner*.

That is the classic failure of optimizing against a proxy. We scored 106,183 candidates on an
idealized free-field model — exact geometry, plane waves, perfect synchronization, no coupling —
and took the maximum. The maximum of `true quality + model error` selects hard on the error
term, so the winner of a large search is exactly where the proxy should be trusted least. Every
array we did *not* select this way still obeys the proxy, which is what proves the point rather
than merely asserting it.

**Consequences.**

- The 35 min exhaustive search was not just wasted, it was actively harmful: it produced the
  worst array of the eight and would have been our recommendation.
- `device3`, chosen because it is a thing you can buy and deploy, beats the searched optimum by
  17 points. Physical reasoning about the deployment beat optimization of a clean-room metric.
- If array selection is wanted, select on *measured* accuracy over a handful of physically
  motivated candidates, or hold out the selection criterion the way one holds out a test set.
- Report the whole sequence honestly. "We ran the search, believed it, and it was the worst
  array we tested" is a stronger contribution than a table of arrays that behaved.

**Two smaller findings in the same table.** `device1_3` at 73.1% beats all 24 microphones at
70.5% with half the hardware, so microphone count is not monotone — though that is one seed
against a ±1.2 pp spread, so it is a tie rather than a win until it is replicated. And `quad4`
puts the four-microphone floor at 54.2%, level with the worst six-microphone arrays, which sets
the bottom of the cost-versus-accuracy curve.

**Caveat on all five new rows:** single seed. The headline grid's ±1.2 pp spread makes gaps of 5
points and up safe, but `device4` against `quad4` (54.5 vs 54.2) is a tie, not a ranking.

## 23. Does our pipeline reproduce the paper? Yes for the baseline, and yes for the model once
the comparison is like-for-like

Their results are in figures, not tables, so this had to be built rather than looked up
(`utils/check_vs_paper.py`). Three conventions had to be adopted before any number is
comparable: their distance bins (0–50 / 50–100 / 100–200 m), the **mean** angular error rather
than our median, and their detection rule, which never thresholds an energy — a frame counts
as detected when the direction estimate is *stable*, defined as the spread of the last three
estimates staying under 10°. Paper values are read off Fig. 5 to the nearest degree.

### The classical baseline reproduces, closely

| Test 2, all 24 mics | 0–50 m | 50–100 m | 100–200 m | Stable-direction false positives |
|---|---|---|---|---|
| Paper, DAS | 38° | 75° | 33° | 67.0% |
| **Ours, DAS** | **40°** | **72°** | **31°** | **68.6%** |

Within a few degrees everywhere, including their odd non-monotonicity where the 100–200 m bin
scores *better* than 50–100 m — we reproduce that too, so it is a property of the recordings and
not of their processing. The false-positive rate lands 1.6 points from theirs.

This is the strongest end-to-end check we have. It exercises microphone geometry, the
per-recorder sample offsets, the STFT, the beamformer, the GPS label alignment and the metric
code all at once, against numbers produced by a different implementation. **No degradation.**

### Our U-Net matches theirs where the data is, once frames are selected their way

| Test 2, all 24 mics | 0–50 m | 50–100 m | 100–200 m | Frames kept |
|---|---|---|---|---|
| Paper, U-Net | 10° | 13° | 14° | not stated |
| Ours, all drone frames | 12–15° | 20–21° | 49–76° | 100% |
| **Ours, stable frames only** | **7–10°** | **9–12°** | 31–74° | ~81% |

Three seeds, range shown. Comparing our all-frames average against their figure was not
like-for-like: their own false-negative rule discards unstable frames, and an error average over
the frames a system was confident about is a different quantity from an average over all of
them. Apply their rule to our predictions and we match or beat them in both bins that hold the
data — 97.5% of drone frames sit under 100 m.

Worth being explicit that this is inference about their protocol, not something they state. What
we can say is that under the one reading that makes their DAS and U-Net numbers mutually
consistent, our model lands on theirs.

**We keep the all-frames average as our primary metric anyway.** Dropping the frames where a
system was uncertain flatters it, and the fraction dropped is itself a result. The paper-matched
row exists for the comparison table and nothing else.

### What we do not reproduce: beyond 100 m

Our error stays high in the far bin where theirs does not. That bin is 374 frames, 2.5% of the
split, and three things about it are worth recording:

- **The drone is not the loudest thing in the map there.** Mean DAS peak energy on far drone frames
  is **138.8 dB against 139.8 dB on no-drone frames**, so a brightest-pixel or mask decoder has
  nothing to lock onto. Stated precisely, because the stronger sentence is tempting and wrong: this
  is a property of the *peak of the beamformed map over 200–2200 Hz*, not of the audio. A harmonic
  comb sitting 10 dB under a broadband floor is invisible to peak energy and still recoverable by a
  matched filter or a comb detector, so "no acoustic evidence" would overstate it. What we measured
  bounds what *our* decoder can do, not what the recording contains.
- The failure is azimuth, not elevation: 53° mean azimuth error against 11° in elevation, with
  predictions scattered across az −40° to 180° while the truth sits in a 132°–170° arc. The
  model keeps the height roughly right and loses the bearing entirely.
- Early stopping monitors `hit_10deg`, which is dominated by the 97.5% of frames under 100 m, so
  nothing in our protocol was ever selecting for far-range behaviour.

Candidate explanations we have not tested: their polar reprojection, Tversky loss, skip-attention
and 64 filters are all things we dropped or changed, and any of them could matter more at low
SNR. Reporting the gap rather than explaining it away.

### Consequence for our own results

Every headline number in sections 21 and 22 used brightest-pixel decoding over all frames,
applied identically to every array and every K, so the comparisons between them are unaffected.
Centroid decoding was tested and changes almost nothing (15° → 15°, 21° → 21° on seed 0), which
also rules out post-processing as the source of the far-range gap.

**Measured cost at `base=32`:** 84 s per epoch at K=1, 165 s at K=5, converging in 19 and 22
epochs. That is 27 min and 63 min per run, so the 18-run headline grid is about 14 h — inside
budget, unlike 40–50 h at `base=64`.

## 24. Per-frame prediction dumps: why summaries were never enough

`metrics.json` holds one number per metric per run, and three questions we still owed the report
cannot be answered from summaries at all: detection against range (a pooled AUROC hides that the
drone stops being the loudest peak past 100 m), temporal integration (needs frames in time order), and
frame-paired comparisons (asking whether K=5 beats K=1 *on the same frame*, which removes the
largest variance term).

`src/predict.py` runs a checkpoint once and writes every per-frame quantity to `results/preds/`:
presence score, both decoders' directions, the argmax cell, the labels, and — the part that makes
the rest possible — `seg` and `frame`, so frames can be re-sorted into time order and grouped by
scene. There are 63 of them for test 2, including the four history depths on the nine K=5 runs.

Three design decisions worth recording:

- **They live under `results/` and are committed**, 20 MB for all 63. They began under
  `cache/preds/`, which was wrong the moment the analyses started depending on them: `cache/` is
  by definition the directory a clone rebuilds from audio, and rebuilding these needs the 4 GB
  STFT cache and a GPU. Committing them is what lets someone with a fresh clone and no dataset
  re-derive every number in the report, not just read it. `savez_compressed` rather than `savez`
  once they became committed artifacts: most columns are grid indices and repeated labels, so
  1.17 MB per dump becomes 0.34 MB, and `np.load` reads either form without being told which.

- **`utils/check_vs_paper.py` now imports it** rather than carrying its own copy of "run the model
  and decode a direction". Two implementations of that would eventually disagree, and the
  disagreement would look like a result.
- **The classical baseline is dumped too.** `src/baseline.py` computed per-frame arrays and threw
  them away, and the audibility analysis in section 25 needs exactly those.

Gate: every causal dump reproduces its own run's `metrics.json` to 1e-6. That check is why the
`summarize()` refactor to accept explicit predicted angles could be trusted — if it had changed
the rectangular path by even a rounding step, 24 runs would have said so at once.

## 25. Detection, stated the way an operator would ask for it

The reference paper reports detection as a stability proxy, and our own reporting had been an
AUROC. Neither answers the operational question, which is "how far, at what false-alarm rate, and
after how long". `utils/detection_range.py` and `utils/temporal_integration.py` answer it from the
dumps, with no retraining.

### Range: about 90 m, and it is a data limit before it is a model limit

Threshold fixed on the no-drone frames alone at a chosen false-alarm budget, then recall measured
per range bin. Test 2 has 3,364 no-drone frames = 5.6 minutes, so **one false alarm is 10.7 per
hour and nothing finer exists**; the budgets below are integer alarm counts, not round rates.

| Recall at 107 false alarms/h | 0–30 m | 30–60 m | 60–90 m | 90–120 m | 120–180 m |
|---|---|---|---|---|---|
| all 24, K=1 | 30% | 32% | 22% | 6% | 2% |
| device 3, K=5 | 36% | 28% | 14% | 1% | 1% |

Per-frame recall of 30% reads as a catastrophe and is the wrong unit. Frames arrive at 10 Hz and a
drone overflies for tens of seconds, so the quantity that matters is the wait for the *first*
alarm: at 30% per frame that is **0.3 seconds**. Under 90 m every array alarms within about a
second; beyond 90 m none of them alarm at all. That is the honest statement, and it is one the
paper never makes.

### Why the far field is unanswerable: the cue is gone, not missed

Using the baseline's absolute beamformed peak energy as the measure (peak-to-mean contrast is at
or below chance as a detector here — AUROC 0.455 on all 24 — so it would be a proxy for nothing):

| Drone frames quieter than a typical no-drone frame | 0–30 m | 30–60 m | 60–90 m | 90–120 m | 120–180 m |
|---|---|---|---|---|---|
| all 24 | 4% | 36% | 57% | 63% | 67% |
| device 3 | 2% | 27% | 52% | 65% | 65% |

Two thirds of the labelled positives beyond 90 m are quieter than typical ambient. We train on
them and score ourselves on them.

**Careful with the word inaudible.** A stricter reading — below the 95th percentile of ambient,
i.e. uncatchable by an energy threshold that fires on 5% of quiet frames — puts 93% of *all* drone
frames in that class, and the network still localizes 70% of them within 10°. So the energy
statistic is what fails, not the frame: there is spatial structure the scalar throws away. Both
numbers are reported for that reason, and neither is called "inaudible" without qualification.

### An unexpected finding: the near field is our worst region

Scoring localization on the loudest 7% of frames — which are almost all under 30 m — gives 54.9%
within 10° against 70.1% overall, and the by-range table shows why: 0–30 m is 67%, 30–60 m is 78%.
Every seed and every array agrees (7.0° median at 0–30 m against 4.5° at 30–60 m). The plausible
cause is that our steering vectors assume a plane wave, which a source a few metres from a 2 m
array is not, so the near field breaks the beamformer's own model. Untested, and stated as a
candidate.

### Integration: time pays for detection even though it did not pay for direction

Sweeping a causal moving average over the presence score, within segments, no retraining:

| all 24, K=1 | window 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| decision latency | 0.0 s | 0.2 s | 0.9 s | 2.9 s |
| AUROC | 0.727 | 0.741 | 0.756 | **0.773** |
| recall at 1 false alarm | 21.6% | 21.9% | 28.5% | **33.7%** |

Monotone in every seed and every array, +0.03 to +0.05 AUROC at 2.9 s of latency. This is a
different mechanism from frame stacking and the distinction matters: stacking hands the network
five raw maps and hopes; integration averages the network's *output*. The presence label is
constant across a half-second window while the direction label is not, so integration averages
repeated measurements of a fixed quantity, which is the textbook case where averaging works.

Two limits kept attached to these numbers. Every segment here is homogeneous, so **no window ever
straddles a drone onset** — a deployed system pays a latency cost exactly there, and this dataset
cannot measure it. And the sweep stops at 3 s deliberately: pushed far enough the window would
cover a whole segment and the task would degenerate into telling two recordings apart.

The seed spread also has to be said out loud. The map model's presence head gives AUROC 0.756,
0.845 and 0.580 on the three all-24 seeds. Integration helps *within* every seed, but the head
itself is a lottery, which is the strongest argument for the separate per-channel detector.

## 26. How deep does history go, and why the null was not what it looked like

The headline grid compared a K=5 model against a separately trained K=1 model and found nothing,
which reads as "time does not help". That reading was wrong, and the experiment that shows it
costs no training at all.

`reorder()` gained a `freeze{j}` intervention: keep the last *j* frames real, fill the earlier
slots with a copy of the oldest kept frame. `j=1` is the old `freeze` control, `j=K` is unmodified
`causal`, and in between the *history depth* varies with the weights, the parameter count and the
input shape all held exactly fixed. Nine existing K=5 checkpoints, five depths each, 36 forward
passes over test 2.

| History depth | j=1 | j=2 | j=3 | j=4 | j=5 |
|---|---|---|---|---|---|
| Within 10° (mean of 9 runs) | 60.4% | 61.8% | 63.2% | 64.1% | **64.7%** |
| Paired increment | — | +1.49 ± 0.21 | +1.31 ± 0.22 | +0.97 ± 0.20 | +0.53 ± 0.07 |

Increments are paired — the same weights at two depths — so the standard error is over runs and
every step is significant. **History is worth +4.3 pp**, and the curve is still rising at five
frames.

### The reconciliation

| Array | K=1 trained | K=5 at j=1 | K=5 at j=5 | Channel cost | History gain |
|---|---|---|---|---|---|
| all 24 | 70.5% | 67.5% | 70.1% | −3.0 pp | +2.6 pp |
| device 3 | 69.3% | 67.6% | 71.1% | −1.6 pp | +3.5 pp |
| best-6 | 52.1% | 45.9% | 52.8% | −6.2 pp | +6.8 pp |

A K=5 model deprived of history is *worse* than a K=1 model trained to use one frame, because
five times the input channels is a real cost in optimization and overfitting. The history then
earns back approximately that cost and no more. So both facts are true at once: **the mechanism
works, and stacking pays for itself and nothing more.** The correct sentence for the report is not
"temporal context does not help" but "temporal context buys back what widening the input costs".

Note also that the history gain is largest on the *worst* array (+6.8 pp on best-6), which is the
first hint of the pattern section 27 confirms.

### Consequence for K=9

The gate condition was "still climbing at j=5", and it is (+0.53 pp, 7 standard errors), so K=9
runs. But the increments are halving each step, so frames six to nine are worth roughly +1 pp of
*benefit* — against a channel cost that grows. The curve measures one side of a trade, and a null
at K=9 would be a statement about the trade rather than about time.

Written before the runs; section 34 has the outcome, which was +2.1 pp rather than +1.0, and the
depth curve repeated on the K=9 weights then locates the saturation this one could only bound.

## 27. Where temporal context pays: exactly where the evidence is weakest

Frame-paired K=5 minus K=1, joined on (segment, frame) so the between-frame variance drops out.
K=5 discards the first four frames of each segment, so this runs on the intersection.

| Within 10°, K=5 − K=1 (pp) | overall | 0–30 m | 30–60 m | 60–90 m | 90–120 m | 120–180 m |
|---|---|---|---|---|---|---|
| all 24 | −0.4 ± 2.0 | −3.2 | +0.1 | +1.8 | +3.5 | −0.1 |
| device 3 | +1.8 ± 1.1 | −0.1 | +3.2 | +3.4 | −1.2 | −3.1 |
| best-6 | +0.6 ± 0.9 | −4.0 | −2.8 | **+8.4** | +6.2 | **+10.6** |

The pooled nulls were hiding a consistent pattern: context **hurts in the near field** and **helps
in the mid and far field**, and it helps most on the array with the least spatial information.
That is what the persistence mechanism predicts — averaging over time is worth most when a single
frame is ambiguous, and worth nothing when it is already unambiguous, where the extra channels
only add capacity to overfit. The near-field loss is consistent with section 25's finding that the
near field is where the plane-wave model is weakest.

Statistical honesty: frames 100 ms apart are strongly autocorrelated, so per-frame disagreement
counts (16–23% of frames disagree between K=1 and K=5) are descriptive only. The error bars that
mean anything are over seeds, and that is what the verdict uses.

## 28. Leave-one-scene-out: the detection result we had been overstating, and by how much

Every detection number in sections 18 and 19 trained on the `train` split and tested on `test_2`.
Listing the segments shows the problem:

    train    Nov_25_2024 pre-takeoff 2.1 min | Mar_18 f1 pre-takeoff 1.3 min | Mar_18 f2 no_drone_1 4.8 min
    test_2   Mar_18 f2 no_drone_2 4.8 min    | Mar_18 f2 pre-takeoff 0.8 min

`no_drone_1` and `no_drone_2` are **two halves of one continuous recording** of the same field on
the same afternoon through the same array. Anything learned about that soundscape transfers to its
own second half for free, so AUROC 0.662 was a within-scene number wearing a generalization label.

Three scenes exist — A = Nov 25 flight 11, B = Mar 18 flight 1, C = Mar 18 flight 2 — so each can
be held out in turn (`utils/detector_loso.py`). Early stopping monitors interleaved 30-second
blocks of the *training* scenes; see section 29 for why that detail decided the result.

| Held-out scene | clean, linear | clean, log64 | + confusers, linear | + confusers, log64 |
|---|---|---|---|---|
| A = Nov 25 (different session) | 0.441 | 0.538 | 0.347 | 0.482 |
| B = Mar 18 f1 | 0.551 | 0.412 | 0.768 | 0.758 |
| C = Mar 18 f2 | 0.472 | 0.582 | 0.662 | 0.715 |
| **mean** | **0.488** | **0.510** | **0.592** | **0.651** |
| sd over folds | 0.056 | 0.088 | 0.219 | 0.149 |

Three findings, in order of how much they change the report:

**1. Cross-scene, the detector is at chance.** 0.488 against 0.662 within-scene. Not "weak" —
*absent*. The honest sentence is that this detector detects the recording, not the drone.

**2. The failure is across sessions, not across flights.** Fold A is the only genuinely
independent site, and it is at or below chance in all four variants (0.347–0.538). Folds B and C
share a session, a site and a calibration with their training data, and they reach 0.66–0.77. The
fold structure was built to expose exactly this difference and it did.

**3. ESC-50 confusers help, and only where anything works at all.** +0.10 clean-to-confusers on
the linear axis and +0.14 with the log axis, concentrated in folds B and C. Mixing outside sound
into training teaches "not every loud broadband thing is a drone" — but it cannot teach what a
different site sounds like, so fold A stays at chance.

For the record, the log-frequency axis of section 31: +0.022 clean and +0.059 with confusers,
against a fold spread of 0.056–0.219. Inside the noise, so the pre-registered decision applies —
notes and one limitation sentence, not the method section.

## 29. The monitor split, which changed the answer twice

The first LOSO run early-stopped on the held-out scene, which is selection on the test set. It was
not a technicality: fold A's held-out AUROC went 0.34, 0.67, 0.72, 0.47, 0.46, 0.80, 0.36 over
seven consecutive epochs. Taking the best of twelve such draws would have reported ~0.80, and a
rerun with a fixed epoch budget would not have reproduced it.

Fix one was to monitor the last 20% of each training segment. That was worse than it looks: the
tail of a flight segment is the far, quiet end of the flight and the tail of a pre-takeoff segment
is the moment before the motors start, so the split paired the hardest positives against the most
drone-adjacent negatives. Fold B's monitor AUROC then sat below 0.4 for most of training, which
says something about the split and nothing about the model.

Fix two, which is what the numbers above use: hold out **interleaved 30-second blocks**, one in
five, with the held block rotating by segment index so no segment is skipped and no segment
donates only its tail. Frames 100 ms apart are near-duplicates, so random frame holdout would
measure memorization; at 30-second blocks only the two frames at each boundary have a neighbour
across the split.

Even with a clean monitor, checkpoint selection is doing visible work. Scoring the held-out scene
with the final-epoch weights instead of the selected checkpoint moves fold C from 0.472 to 0.662
and fold A from 0.441 to 0.252. `metrics.json` now stores both, and the gap belongs in the
report as the error bar on this protocol.

## 30. DEMAND: the false-alarm rate is a wall, not a number

Ninety minutes, 18 environments, 48 kHz — our own sample rate, so no resampling — single channels,
which is why the array mismatch does not matter for a detector that reads one microphone at a
time. Each scene is rescaled so its median in-band frame energy matches our own ambient's, since
DEMAND's absolute level reflects its own preamp and says nothing about how loud that place would
be through our array.

The threshold is fixed on **our** drone frames at a stated recall and never touched afterwards.

| Threshold set at | our ambient fires | DEMAND fires | DEMAND alarms/hour |
|---|---|---|---|
| 95% drone recall | 93% | 69% | 24,881 |
| 90% | 85% | 56% | 20,113 |
| 75% | 60% | 34% | 12,099 |
| 50% | 25% | 19% | 6,733 |

The plan for this stage was to take the resolvable false-alarm floor from 10.7 per hour down to
0.7. The floor turned out to be irrelevant: at every usable operating point the detector fires on
tens of percent of all ambient — 6,733 alarms per hour even at 50% recall, which is roughly two
per second. There is no threshold at which this is a detector.

Two things the measurement adds beyond that:

- **It is not our recordings.** Our own ambient and 18 unrelated soundscapes give the same answer,
  so nothing about our field, our microphones or our 5.6 minutes is responsible.
- **The spread across scenes is small**, 23.8k to 30.8k per hour with the confuser model
  (median 26.2k, IQR 24.7–29.3k), and outdoor scenes are only slightly worse than indoor. A
  detector that fires on everything has no interesting variation across environments, which is
  itself the diagnosis.

Caveat kept attached: level-matching applies +47 to +80 dB of gain to these recordings, so their
own noise floor is amplified along with the scene. The number is indicative of a saturated
detector, not a calibrated deployment figure.

## 31. Why a log-frequency axis should have helped, and the honest null

A multirotor's sound is a harmonic comb: the blade-pass frequency (rotor RPM over 60, times blade
count — a few hundred hertz for an Air 3) with energy at every integer multiple through our band.

The comb moves with throttle. Ten percent more RPM scales every peak by 1.1, which on a linear
frequency axis is a *different* shift in hertz for each harmonic — the tenth moves ten times as
far as the first. The comb stretches rather than slides. Convolution is invariant to sliding, not
to stretching, so a linear-axis network has to learn the low-RPM comb, the mid-RPM comb and the
high-RPM comb as separate patterns, spending parameters and examples on what is physically one
pattern in different states.

On a log axis multiplication becomes addition: at position `log f`, scaling frequency by any
factor translates every peak by the same distance. The comb slides rigidly, its shape is
independent of RPM, and one template covers every throttle setting. Nothing is added to the
model — a variation it currently has to memorize is removed, which is the right kind of change
when the ceiling is data rather than capacity.

Implementation is one fixed matrix multiply: a triangular filterbank, mel-style but log-warped,
applied to the magnitude spectrum before the convolutions. Pooling happens on linear magnitude
before the decibel conversion, because summing energy and then taking the logarithm is what a
filter does; taking the logarithm first would average decibels, a geometric mean in which one
dead bin drags a whole band down. No audio re-read and no cache rebuild.

Bin count is set by a real constraint rather than a guess. The cache holds 381 linear bins at
10 Hz spacing, and at the bottom of the band 10 Hz is 5% of 200 Hz, so log bins narrower than
about 5% would interpolate rather than measure. Over 200–4000 Hz that caps us near 64 bins, which
is a bonus: the input shrinks from 381 to 64 and the model gets smaller and faster.

**Result: null.** Under the section 28 protocol, +0.022 AUROC clean and +0.059 with confusers,
against a fold spread of 0.056 to 0.219. The pre-registered reading applies — if the detector is
keying on broadband energy rather than comb structure, the axis cannot matter, and everything
else in sections 28 and 30 says that is exactly what it is doing. The argument is recorded here
because it is sound and because the null is informative about what the model learned; it earns
one sentence in the report's limitations and no place in the method.

## 32. The polar disc: the right hypothesis, and it was wrong

This was the one ablation aimed at a failure we had already localized. Beyond 100 m our error is
almost entirely azimuth — 53 degrees against 11 in elevation — and azimuth is the axis a
rectangular elevation-by-azimuth map distorts. That map is a plate carree projection of the sky:
near the zenith one azimuth cell covers almost no solid angle, so a single 3x3 kernel means
different things at different elevations. Circular padding fixed the seam at 0/360 degrees years
of literature ago; nothing fixed the pole.

The disc puts the zenith at the centre and the horizon on the rim, which is what a fisheye lens
does. A kernel then means the same thing everywhere, and azimuth resolution is *highest* at the
rim — exactly where a distant, low-elevation drone sits. So the hypothesis was specific and the
prediction was directional.

Implementation as a preprocessing flag rather than a code fork, per notes section 7: one
precomputed `(46, 46, 2)` sampling grid and an `F.grid_sample` for the input, the same remap
applied to the target mask so input and target cannot drift apart, and a centroid decoded in disc
coordinates and inverted analytically back to azimuth and elevation. `wrap=False` on the
convolutions, since the disc has no seam to wrap.

Two gate failures worth recording, both caught before any accuracy number was read:

- **Dead corners were not dead.** The disc is inscribed in a square, so 21.5% of the tensor is
  outside the sky and must be exactly zero. `grid_sample` with `padding_mode="zeros"` still
  produces small non-zero values just outside the rim by bilinear interpolation against real
  samples, so the output is now multiplied by the validity mask explicitly.
- **The round-trip test was too weak.** It used a flat cap as the target, which decodes to the
  right centroid even under a badly wrong remap. Replaced with a Gaussian-like beam, so the test
  is sensitive to the peak's *location* rather than to the blob's symmetry.

A third bug survived the gates and was caught only because the first three runs crashed: the
training loop was feeding rectangular maps against polar targets, because `to_maps()` was called
with the projection in the evaluation path and without it in the training path. The crash was a
shape mismatch in the loss, which is the good kind of bug — it cannot be mistaken for a result.

**Result, all24 at K=1, three seeds each, mean over seeds:**

| Mean error (deg) | 0–50 m | 50–100 m | 100–200 m | within 10 deg |
|---|---|---|---|---|
| rectangular | 11.2 | 19.8 | 57.0 ± 16.0 | 70.1% ± 3.6 |
| polar disc | 12.1 | 22.0 | 82.4 ± 12.9 | 71.0% ± 1.8 |

Azimuth error, the quantity the stage was run to move. No overall change (71.0 against 70.1,
inside the seed spread) and the far bin is 25 degrees *worse* against a 16-degree spread — inside
two standard deviations, so formally a null, but with no hint of the predicted gain. The
projection was not what the far-field failure was.

In hindsight the answer was already in section 25: beyond 90 m two thirds of the labelled drone
frames are quieter than a typical no-drone frame. No projection recovers a cue that is not in the
audio, and a resampling that costs 21.5% of the tensor to dead corners has to pay for itself out
of something. Per the plan, polar does not go on to `device3` and does not go into test 1.

## 33. Sub-band count: a documented guess becomes a documented insensitivity

Sixteen sub-bands over 200-2200 Hz was our reconstruction of a detail the paper does not specify
(notes section 16). Three runs on `device3` at K=1 settle what that guess cost:

| Sub-bands | 8 | 16 | 32 |
|---|---|---|---|
| Within 10 deg | 69.5% | 70.0% | 69.9% |
| Median angular error | 5.54 deg | 5.68 deg | 5.32 deg |

A 0.5 pp spread against a 1.2 pp seed spread: the choice does not matter, which is the useful
outcome. It means the reconstruction is not load-bearing, and it also says something about the
representation — if eight bands carry as much as thirty-two, the network is using the
*consistency* of a direction across the spectrum rather than fine spectral detail, which is the
same conclusion the order-invariance result reached from the time axis.

One asymmetry worth noting: the presence head's AUROC on these same three runs is 0.775, 0.562
and 0.734. Localization is insensitive to the band count and detection swings by 0.2, which is
the presence head being a lottery again rather than the band count mattering.

## 34. K=9: the gate made a prediction, and the prediction was too pessimistic

Stage A4 existed to decide whether K=9 was worth four hours of compute. It measured the benefit of
history at a *fixed* input width — the same trained K=5 weights evaluated at depths 1 to 5 — found
+4.3 pp still climbing at j=5, and extrapolated the decelerating increments to roughly +1.0 pp for
frames six to nine. That was the justification: not "more is better", but a slope we had measured.

**Trained result, `device3`, three seeds:**

| Cell | K=1 | K=5 | K=9 |
|---|---|---|---|
| Within 10 deg | 69.3% ± 0.7 | 71.1% ± 1.8 | **73.2% ± 1.4** |

+2.1 pp over K=5 and +3.9 over K=1, against our ±1.2 pp yardstick, and monotone in K. Two things
follow that we did not have before:

- **Temporal context does help localization**, on this array, at this range distribution. The
  headline grid's null at K=5 was a net of two opposing effects (notes 26), and nine frames is where
  the history term finally outruns the channel term.
- **A cheap recorder with 0.9 s of history beats 24 synchronized microphones with one frame**
  (73.2% against 70.5%). We never trained all24 at K=9, so this is *not* "six microphones beat
  twenty-four"; it is "0.9 s of latency substitutes for 18 microphones", which is the trade a
  deployment actually faces, since one is a hardware cost and the other is a delay.

**And now the curve terminates.** Repeating the A4 intervention on the K=9 checkpoints, depths 1 to
9, gives eight consecutive significant increments and then nothing: +1.12, +0.79, +0.89, +0.72,
+0.47, +0.42, +0.26 and finally +0.19 ± 0.11 pp from j=8 to j=9, the first step inside noise.
History saturates at about 0.8 s, which is now *measured* rather than assumed — the thing A4 could
only bound from below. There is no K=13 to run: the last frame we added was already free of
information, and the answer would be a statement about channel cost rather than about time.

The physical reading is consistent with notes 20. The usable cue is persistence, not motion — a
drone stays put while sidelobes flicker — and averaging persistence over *n* frames improves as the
square root of *n*, so 1 to 9 is a factor of 3 and the increments must decelerate exactly as they
do. What the curve adds is where the useful window ends, and 0.8 s is a number a deployment can
budget for.

## 35. Test 1, scored once: what a sealed split actually bought us

Three minutes, 1,592 drone frames, a different site and date with its own array calibration, held
out since the first commit. `utils/unseal_test1.py` scores it from existing checkpoints, writes
`results/test_1.json`, and refuses to overwrite that file without `--force`. It also **gates what
gets scored**: `polar_won()` and `k9_won()` re-read the test-2 comparisons and admit a variant only
if it beat its counterpart by more than the larger seed spread. Polar failed the gate and was never
scored on test 1; K=9 passed and was. Without that, "score everything we built" quietly becomes a
search over test 1.

**What it said.** Every U-Net cell lost 15 to 18 points against its test-2 figure and the seed
spread went from ±1.2 pp to ±5–8 pp, so no cell here separates from another. Three results survive
that noise:

| Test 1 | within 10° | 0–30 m | 60–90 m | 90–120 m | 120–180 m |
|---|---|---|---|---|---|
| DAS, 24 mics | 63.9% | 5.9 | 3.7 | 36.3 | 58.7 |
| U-Net all24 K=1 | 52.2% ± 5.5 | 13.3 | 5.6 | 5.6 | 40.5 |
| U-Net device3 K=5 | 56.7% ± 7.6 | 15.1 | 6.6 | 5.3 | 27.2 |
| U-Net device3 K=9 | 52.0% ± 3.2 | 20.4 | 6.5 | 6.6 | 21.2 |

1. **The headline claim replicates.** `device3` ≥ `all24` at both K, on hardware costing a quarter as
   much, at a site neither model had seen. This is the one number here we would defend.
2. **The classical baseline wins the pooled comparison**, which our test-2 numbers gave no reason to
   expect: DAS is *better* at this site than at ours inside 90 m (5.9° and 3.7°, against a
   training-site baseline that degrades 8.6× over the same span) while still collapsing past it.
   Two thirds of test 1 sits inside 90 m, so the pooled number goes to DAS. Our model keeps its
   advantage exactly where it always had one — flat at 5–6° out to 120 m, where DAS reads 36°.
3. **The range-conditioned prediction from test 2 held in shape.** Notes 27 said deep history helps
   where evidence is weakest and costs something up close. Comparing within `device3`, so the array
   is held fixed: K=9 has that array's best 120–180 m error (21.2° against 25.0° at K=1) and its
   worst 0–30 m error (20.4° against 14.8°). Sixty-seven frames in the far bin make this suggestive
   rather than significant, and K=5 sits above K=1 there (27.2°), so the ordering is not monotone —
   but the sign of the trade is the one the frame-paired analysis predicted, which is a stronger
   check than matching a pooled mean.

**What we would do differently.** The near field is now our clearest open problem: worst on test 2,
worst on test 1, and the thing that fails to transfer. It is also the most geometry-sensitive regime
— a source metres from a 2 m array violates the plane-wave steering assumption — and this session
carries its own calibration, so the two suspects are confounded and neither is testable without
another site. The honest position is that we named the weakness before unsealing, and the sealed
split agreed with us about it.

One protocol note for whoever comes next: the detector's 0.639 AUROC here is *not* reassurance after
the leave-one-scene-out collapse (notes 28). Test 1's negatives are one minute from one scene, which
is the same single-soundscape flaw at a smaller sample size. A detector cannot be validated on this
dataset at all, and that limitation is a property of the data rather than of the model.
