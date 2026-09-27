# Acoustic-Drone-Detection

Hostile drones are a growing security threat, including on Israel's northern front. Radio
detectors can fail when links are weak, jammed, encrypted, or missing. Acoustic detection is
a useful backup for early warning from sound alone.

Deep learning course project, 2026b. Dan and Shai.

## What we are actually claiming

We detect a drone and estimate its direction from a microphone array, and we ask one
question the reference paper left open:

> **Can temporal context let a cheap 6-microphone array match a 24-microphone one?**

**Measured answer: yes, and it does not even need the context to draw level.** The cheap array
matches all 24 at one frame each — 69.3% ± 0.7 against 70.5% ± 1.5 of frames within 10°, inside the
±1.2 pp seed spread — and with nine frames of history it goes past it, to **73.2% ± 1.4**. We then
gave all 24 the same nine frames, which is the obvious objection, and it does not move: 70.1% ± 2.7
against 70.1% at K=5 and 70.5% at K=1. So **depth pays only where microphones are scarce**, and the
claim is *0.9 s of latency substituting for 18 microphones, adding nothing once you have them* —
the trade a deployment actually faces. History then stops paying: re-scoring those weights at 1 to 9
frames gives eight significant increments and a ninth inside noise, so the useful window is about
0.8 s, measured rather than assumed (notes 26, 34, 37).

This is also the only claim here that survives multiple-comparison correction. Deleting the history
of the trained K=9 model costs 5.1 pp with a cluster-robust 95% interval of [3.6, 6.8], and the gain
lands where one frame is worst: +0.5 pp inside 30 m against +21.8 at 90–120 m. Of fourteen claims
tested together under Holm-Bonferroni, two clear 0.05 and both are this one (notes 40, 41).

The surprise is the third array: the subset our own search called optimal is 18 points *worse* than
a single off-the-shelf recorder. Thirteen arrays later, the reason is not the hardware but the
search — the beampattern criterion predicts accuracy at −0.80 across every array it did not
choose, and fails only on the winner it selected out of 106,183 candidates. Design notes
sections 21 and 22.

**Two findings that came out of interrogating our own numbers, not out of new models.** The pooled
"context does nothing" at K=5 is a *net*: deleting the history of an already-trained model costs
4.3 pp, and widening the input to five channels costs about what history returns, so the mechanism
always worked and only stopped hiding at K=9 (notes 26–27, 34). And our detection result did not
survive its own protocol — the negatives are one recording split in half, so scoring across *scenes*
instead drops detection to chance, and 90 minutes of real ambient from DEMAND puts the false-alarm
rate at 21,591 ± 5,532 per hour (notes 28–31). Localization is supervised by geometry and we have
plenty; detection is supervised by scenes and we have three. Two later attempts to fix it both
failed in the same shape: training on DEMAND's own ambient cuts the rate 4.5× on the scene groups it
heard and only 2.4× on the held-out one, and giving the detector temporal context raises AUROC on
our site while making false alarms on unseen ambient *worse* (notes 38, 39).

**And test 1, the sealed site, is now spent.** Scored once, at the end: the cheap array is again no
worse than all 24 (54.4% ± 4.6 against 52.2% ± 5.5), everything degrades at the new site, the seed
spread quadruples to ±5–8 pp, and classical beamforming *wins* the pooled number there while still
collapsing beyond 90 m where our model holds 5–6°. Report section 7.1 has the range breakdown.

Fewer microphones means a spatially aliased beamformed map — the array invents phantom
peaks that a single frame cannot distinguish from the real source. Phantoms move between
frames while a real drone moves smoothly, so several consecutive frames may recover what
one frame cannot. The reference paper diagnoses exactly this failure and names temporal
context as future work, never testing it, and never on its own 24-channel array.

The headline comparison is **device 3 (one recorder, six microphones, no inter-device
synchronization) against all 24**, at 1 frame versus 5 frames of context.

## Documents, in reading order

| File | What it is |
|---|---|
| [docs/report.md](docs/report.md) | **The final report, sealed.** Every number in it is measured, including the sealed split. Five pages of the five allowed, one four-panel figure, four tables. Section 8 has the division of work |
| [docs/prereg_test1.md](docs/prereg_test1.md) | Eleven numbered predictions with explicit confidences, written before the last round of runs existed, and scored afterwards. Four came back plainly wrong — including both directions of the polar question and the expectation that all 24 would gain from K=9 — which is the point of writing them down |
| [docs/proposal_uav_acoustic_seld.md](docs/proposal_uav_acoustic_seld.md) | The submitted proposal (PDF built with `utils/build_pdf.sh`) |
| [docs/review_plan.md](docs/review_plan.md) | A four-phase review of the pipeline (preprocessing, dataset, training, evaluation): findings, the improvement plan, and a status table of what has been done |
| [docs/design_notes.md](docs/design_notes.md) | Everything measured, decided, or deliberately deferred. The reference document — sections 1–52, including the microphone-subset search, the full experiment grid, the leave-one-scene-out detection result, the ablations that returned informative nulls and the one we later withdrew, where history saturates and where its gain lands, how the significance testing was done, what the sealed split said, and the Track B retrain (notes 47–52) |

If you read one thing, read the report. If you read two, add design notes sections 21 and 22:
what the 18 headline runs say, and why the array our own search called optimal came last.

## What is in the repository

The trained models are here. A clone is enough to load one and reproduce every number in the
report; the 24 GB of audio is only needed to train a new run or score a split we never dumped.

| Committed | Size | What it gives you |
|---|---|---|
| `runs/*/metrics.json`, `config.json` | 2.3 MB | Final metrics **and** the full per-epoch history for all 81 runs, plus the exact config each was trained under, including the git SHA |
| `runs/<id>/best.pt` for 28 runs | 188 MB | **Loadable weights.** All 18 headline runs — `all24`, `device3`, `best6` at K=1 and K=5, three seeds each — plus `device3` and `all24` at K=9, three seeds each, and the four full-data detectors |
| `results/preds/*.npz` | 33 MB | Per-frame output of 123 dumps, 105 on test 2 and 18 on test 1: presence score, both decoders' directions, labels, `seg` and `frame`. Every per-frame analysis reads these, so they run with no dataset and no GPU |
| `results/*.json`, `figures/` | 1.5 MB | The finished tables, the report's figure, and the per-stage diagnostic plots |

**All three seeds, not one per cell**, because the headline claim is itself a statement about
seed spread: device3 at 69.3% ± 0.7 against all24 at 70.5% ± 1.5, a gap inside the noise. Seed 0
happens to be device3's best and all24's worst, so anyone loading a single checkpoint per cell
would measure device3 *beating* all24 and conclude something stronger than we claim. Quote a
spread, ship the spread.

Deliberately **not** committed: `data/` (24 GB of audio), `cache/` (the 4 GB STFT cache, which
`src/stft_cache.py` rebuilds in ~4 min), `logs/`, and 163 MB of other checkpoints — the
coherence-test arrays, the LOSO folds, the base=64 probes. Those are single runs whose metrics
are in `metrics.json` and whose per-frame output is in `results/preds/`, so nothing in the report
depends on having their weights.

## Running the trained models

Both models are committed and neither needs retraining. Every command and snippet below has been
run and verified against this repository, with its output quoted where it is short enough.

### Step 0 — environment (always)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Step 1 — the numbers, with no dataset at all

The committed per-frame dumps in `results/preds/` mean every table and figure rebuilds in about a
minute, no audio and no GPU:

```bash
.venv/bin/python utils/summarize_grid.py        # headline grid, order control, array comparison
.venv/bin/python utils/check_history_depth.py            # history depth over the K=5 runs
.venv/bin/python utils/check_history_depth.py --K 9 --subsets device3   # ...and where it saturates
.venv/bin/python utils/history_by_range.py      # where that history gain lands, by distance bin
.venv/bin/python utils/significance.py          # block bootstrap, across-seed tests, Holm over all
.venv/bin/python utils/compare_polar.py         # rectangle against the disc, resampled and native
.venv/bin/python utils/detector_history.py      # the detector at K=1/5/9, AUROC against false alarms
.venv/bin/python utils/check_vs_paper.py        # our numbers against the reference paper's
.venv/bin/python utils/plot_report_figures.py   # the report's figure, from results/*.json
.venv/bin/python utils/confusion.py             # confusion matrices at the 95%-recall threshold
.venv/bin/python utils/calibration.py           # presence-head reliability diagram
.venv/bin/python utils/error_structure.py       # direction error by elevation, signed azimuth shape
.venv/bin/python utils/persistence_detector.py  # direction stability as a detection score
.venv/bin/python utils/hybrid_decoder.py        # delay-and-sum / U-Net per-frame gate
```

`significance.py` is the one to read first if you want to know which of these numbers we are willing
to defend: it prints every claim with its interval, marks the ones whose cluster count is too small
to test, and applies Holm-Bonferroni across the whole family.

### Step 2 — the dataset and the STFT cache (needed to *run* either model)

Nothing runs a model on audio without the STFT cache, and the cache is built from the dataset.
If you already have the dataset, put it under `data/` in exactly this layout — `src/config.py`
addresses these four session directories by path, so the names matter:

```
data/training/Nov_25_2024/flight_11/     20*.wav (4 recorders), *-Airdata.csv, alignment_params.json
data/training/Mar_18_2025/flight_1/      same
data/test_2/Mar_18_2025/flight_2/        same
data/test_1/Oct_11_2024/flight_5/        same
```

If you do not have it, `utils/download_dataset.sh` fetches it (24 GiB, ~1 h, resumable, safe to
re-run, skips the 19 GB of 360° video). Then:

```bash
.venv/bin/python -m src.stft_cache            # ~4 min, writes cache/ (3.4 GiB)
.venv/bin/python utils/check_stft_cache.py    # prints GATE PASS if the cache is sound
```

`data/` and `cache/` are git-ignored, so this step is the one thing a clone cannot skip.

### Step 3 — run the localization model (U-Net over beamformed maps)

Pick any of the 18 committed run IDs, `{all24,device3,best6}_K{1,5}_w32_b16_wce_s{0,1,2}`:

```bash
.venv/bin/python -m src.predict --split test_2 --runs device3_K1_w32_b16_wce_s0
```

That writes `results/preds/<run>_test_2_causal.npz` and prints `GATE PASS` when the run's own
`metrics.json` is reproduced to 1e-6. Drop `--runs` to do every run, add `--split test_1` for the
held-out site, `--overwrite` to recompute a dump that already exists. From Python:

```python
from src import predict
run = "device3_K1_w32_b16_wce_s0"                    # the cheap 6-mic array, the headline claim
pred = predict.cnn_predictions(run, split="test_2")  # per-frame arrays: score, pred_az, pred_el, ...
assert predict.verify(pred, run) == []               # [] means it matches metrics.json
```

Note that `cnn_predictions` returns the committed dump when one already exists, so on test 2 this
works even *without* step 2. Pass `overwrite=True`, or ask for a split or run with no dump, and it
needs the cache.

### Step 4 — run the detection model (per-channel spectral detector)

This one has no CLI, because it is always scored as a whole split. Four checkpoints are committed:
`detector_K1_w32_s0_full` and the `_log64`, `_conf`, `_conf_log64` variants.

```python
from src import detector
from src.config import CACHE_BAND, MIC_SUBSETS
from src.data import SplitData

model, saved, stats, bank, dev = detector.load_run("detector_K1_w32_s0_full")
data = SplitData("test_2", MIC_SUBSETS["all24"], saved["config"]["K"],
                 band=CACHE_BAND, verbose=False)
m = detector.evaluate(model, data, stats, dev, bank=bank)
print(m["auroc"], saved["test_2"]["auroc"])          # 0.6624 0.6624
print(saved["test_2"]["threshold_at_95_recall"])     # 0.0844, the operating point we quote
```

Takes ~5 min on CPU and well under a minute on MPS — it is not hung. `bank` is the log-spaced
filterbank, rebuilt from the config, and it must be passed for the `_log64` checkpoints or the
input shape will not match. `utils/demand_false_alarms.py` is the worked example of pointing this
detector at outside audio it never saw.

### Training from scratch, if you want to

```bash
.venv/bin/python -m src.train --subset device3 --K 5 --seed 0   # one localizer run, ~45 min
.venv/bin/python -m src.detector --K 1 --width 32               # one detector, minutes
```

Whole batteries go through `utils/run_grid.py`, which is sequential (MPS is one compute unit, so
two runs at once each take twice as long) and skips any cell already on disk, so it is safe to
re-run and safe to interrupt. The batteries behind the report, in the order they were run:

```bash
.venv/bin/python utils/run_grid.py headline    # the 18 headline runs, ~14 h
.venv/bin/python utils/run_grid.py all24k9     # closes the K=9 gap in the grid, ~4 h
.venv/bin/python utils/run_grid.py tversky     # the reference paper's loss, on the rectangle
.venv/bin/python utils/run_grid.py polarnative # the disc, steered at its own cells
.venv/bin/python utils/run_grid.py polarnative_tversky   # both at once
.venv/bin/python utils/demand_trained.py       # six leave-one-group-out detector folds, ~1 h
```

Run `utils/check_polar.py` before any disc battery. Every way the disc can be wrong is silent — a
convention mismatch, a half-cell offset, or a memoized rectangular steering tensor handed to a disc
run — and all of them train happily to a wrong answer. The gate also measures the mechanism the
disc rests on, that native steering peaks more sharply than resampling, before hours of compute
assume it. Set `RUN_GRID_STOP_HOUR` to leave a queue unattended: it is checked before every
individual cell, so the queue stops at a known hour rather than mid-battery.

### Two things that will waste your afternoon

- **`best.pt` is a bare `state_dict`.** The architecture and the input normalization constants
  (`input_mean`, `input_std`) live in the JSON next to it — `config.json` for the localizer,
  `metrics.json["config"]` for the detector, which never writes a `config.json`. Load through
  `predict.cnn_predictions` or `detector.load_run` and this is handled for you. Call `torch.load`
  yourself, skip the normalization, and you get a model that silently scores garbage.
- **Not on a Mac? Pass `device` explicitly.** Every `config.json` records the device its run was
  trained on, which here is `mps`, and that recorded value is the default — `pick_device` returns
  what it is asked for without falling back. So on a Linux or CUDA machine use
  `cnn_predictions(run, split, device="cpu")`, `detector.load_run(run_id, device="cpu")`, or
  `--device cpu` on the `src.predict`, `src.train` and `src.detector` CLIs. Otherwise you get an
  MPS error on a machine that has no MPS. Relatedly, run anything touching the GPU from a normal
  shell: a sandboxed or restricted shell reports a bogus "MPS requires macOS 14.0+" error even on
  hardware that supports it.

**Test 1 was sealed, and is now spent.** Every choice in this repository was made on test 2.
`utils/unseal_test1.py` scored the held-out site once, at the end, from checkpoints that already
existed, and wrote `results/test_1.json`; it refuses to overwrite that file without `--force`, and
nothing in the repository has been tuned since. If you extend this work, treat those numbers as
read-only and go back to test 2. The dataset is the Western Sydney University UAV Acoustic
Localization Dataset (CC BY-SA 4.0, no registration).

## Progress

| Step | State | Gate result |
|---|---|---|
| Dataset download | **done** | 28 files, 24.13 GiB, all byte-verified |
| Microphone subset search | **done** | Every non-coplanar 4- and 6-subset scored on 250 directions, 35 min |
| 0. Environment | **done** | `requirements.txt` pinned to installed versions |
| 1. Geometry and labels | **done** | Reproduces 1.6° median azimuth, 5.1° elevation, 91% within 10° |
| 2. STFT cache | **done** | 50,474 frames, 3.44 GiB, 215 fps; cached frames still steer to 1.6° |
| 2b. Subset scoring in `src/arrays.py` | **done** | Device 4's mirror response is exactly 1.00000 of its true peak, as coplanarity demands; all 11 pinned subsets reproduce their cached scores |
| 3. Classical baseline and metrics | **done** | Angular error grows 8.6× from the near field to 60–90 m. See design notes section 17 — device 3 beats all 24 here, which reframes the claim |
| 4. Single-frame CNN | **runs**, gate split | Localization passes hard: 6.2° median against the baseline's 22.1°, and the range degradation is gone. Detection does not pass — the dataset has only 4.8 min of negatives. See design notes section 18 |
| 4b. Per-channel detector | **done**, negative result | 86k parameters over one microphone's spectrum ties the 7.9 M map head, 0.662 against 0.669 AUROC. Detection is bounded by how many *scenes* we have, not how many examples. Kept anyway: it is the only clean way to admit outside mono audio. Design notes section 19 |
| 5a. Frame stacking plus order control | **done** | Gate passes on the mechanism: freezing the stack to the current frame costs 4.7 pp, so the model does look back. Order is irrelevant (0.2 pp), the predicted persistence result. Replicated across all nine K=5 grid runs at 4.3 pp and 0.1–0.2 pp. Design notes sections 20 and 21 |
| 5b. Headline grid, 18 runs | **done** | 808 min, no failures. Six microphones on one recorder match all 24 (gap inside seed noise at both K). Temporal context: +1.8 pp on device3 against a ±1.2 pp seed spread, null elsewhere. The searched-optimal `best6` loses by 18 pp. Design notes section 21 |
| 6a. Coherence test, 5 runs | **done** | 176 min. Killed our own clock-sharing explanation: arrays spanning two recorders do fine, and one of them is the best array we have. The real story is selection on a proxy — the beampattern score predicts accuracy at −0.80 for every array it did not choose, and fails only on its own winner. Design notes section 22 |
| Paper reproduction check | **done** | Our DAS lands within a few degrees of theirs in every distance bin (40/72/31° against 38/75/33°) and their stability-based false-positive rate reproduces at 68.6% against 67.0%. Our U-Net matches theirs under 100 m once frames are selected by their own rule. Beyond 100 m we do not reproduce them; that bin is 2.5% of frames and the drone sits below the ambient there. Design notes section 23 |
| History-depth curve | **done** | The same nine K=5 checkpoints re-scored at 1 to 5 frames of real history: +4.3 pp, monotone, every increment significant. Reconciles the pooled K=5 null — five channels cost 3.0 pp, history earns back 2.6. Design notes sections 26 and 27 |
| Detection under leave-one-scene-out | **done**, negative result | Our negatives are one recording split in half; scored across *scenes* instead, detection is 0.488 ± 0.056 AUROC, at chance. DEMAND's 90 minutes of real ambient puts the false-alarm rate at 24,881/h at 95% recall. Design notes sections 28-31 |
| Polar map, sub-band sweep, log-frequency detector | **done**, three nulls, one later withdrawn | Each aimed at a diagnosed weakness and each declined to move: disc projection 71.0% ± 1.8 against 70.1% ± 3.6, band count 0.5 pp across 8/16/32, log axis inside the fold spread. The disc null turned out to be our own resampling — see the native-steering row below. Design notes sections 31-33 |
| K=9, three seeds | **done**, and it wins | 249 min. 73.2% ± 1.4 against 71.1% ± 1.8 at K=5 and 69.3% ± 0.7 at K=1, monotone in K and past the ±1.2 pp yardstick, so temporal context does pay for localization — it just needed nine frames. Re-scoring the same weights at 1 to 9 frames puts saturation at about 0.8 s. Design notes section 34 |
| Test 1 unsealed | **done**, once | The cheap array holds its parity (54.4% ± 4.6 against 52.2% ± 5.5), everything drops, the seed spread quadruples, and DAS wins the pooled number while still collapsing past 90 m. Report section 7.1 |
| `all24` at K=9, three seeds | **done**, and it is a null that matters | 70.1% ± 2.7 against 70.1% at K=5 and 70.5% at K=1. Closes the one gap in the grid and turns the K=9 result into the sharper claim: depth substitutes for microphones and adds nothing on top of them. Design notes section 37 |
| Native disc steering, ± Tversky loss | **done**, retracts a null | The first disc resampled the finished rectangular map; steering the beamformer at the disc's own cells instead costs the same and cuts far-bin azimuth error from the rectangle's 57.0° to 48.3°, or 45.9° with the reference paper's Tversky loss. Declared a *win in point estimate only*: that bin holds two continuous crossings, so the honest test gives p = 0.43–0.50. Design notes section 36 |
| Detector trained on DEMAND, leave-one-group-out | **done**, negative result | Real ambient does help, but only where it has been: 4.5× fewer false alarms on the five groups it trained on against 2.4× on the sixth. It learns soundscapes, not what a drone is not. Design notes section 38 |
| Detector with temporal context, K=1/5/9 | **done**, negative result | AUROC on our own site rises 0.691 → 0.752 → 0.785 while false alarms on unseen DEMAND ambient rise 21.6k → 27.1k → 29.0k per hour. Context sharpens the boundary at the site it was trained on and does not transport it. Design notes section 39 |
| Statistical significance pass | **done** | Cluster-robust block bootstrap for within-weights claims, Welch and exact permutation across seeds, Holm-Bonferroni over all fourteen. Two survive, both the history intervention; a design effect of 3.5 says frame-level intervals were 3.5× too narrow. Design notes section 40 |
| Where the history gain lands, by range | **done** | +0.5 pp inside 30 m, +8.7 at 60–90 m, +21.8 at 90–120 m, monotone in depth in the far bin. Temporal integration buying signal-to-noise on faint targets, which is what the order-invariance already implied. Design notes section 41 |
| Native disc at K=9, the last untested cell | **done**, negative result | The disc's far-field gain does not transfer to the array and depth where history pays: 62.7° of far-bin azimuth error against the rectangle's 56.5°, pooled accuracy unchanged. The two far-field mechanisms attack one failure, so whichever runs first takes what is available. Design notes section 42 |
| Review pass, no-dataset analyses | **done** | At 95% recall every model fires on 72–96% of empty-sky frames, and the presence probabilities are saturated (Brier worse than a constant). Direction accuracy depends on elevation within every range bin; the far field is lost, not biased. Every headline cell carries a ±5.5 pp interval, so the table separates only `best6`. The U-Net's direction *stability* detects at AUROC 0.81–0.86 on test 2, above its own presence head and steadier across seeds. A delay-and-sum/U-Net gate is a null, but it located a near-field failure unique to `all24`. Design notes sections 43–46 and the section 40 addendum |
| 6b. Remaining ablations | **done** (Track B) | New baseline 69.9% ± 0.9 on a true test 2. No flag beat that by more than seed spread. LOFO reprints March and drops November to 24–28%. Test 1 not scored again. Notes 47–52. Speed head still unused |

## What is left

The report is sealed. Course work outside this repository:

| # | Task | Cost | Why it matters |
|---|---|---|---|
| 1 | Slides, 15 minutes | — | Graded separately |
| 2 | Optional: the speed head from the unused GPS velocity labels | ~1 h | Depth, not a gap. Everything the report claims is measured without it |

The review track ([docs/review_plan.md](docs/review_plan.md)) is finished:

| # | State | Note |
|---|---|---|
| B1 | **done** | Labels lead audio ~0.4 s; recorder 4 is 6–8 samples late (notes 47) |
| B2 | **done** | Hann did not raise hit@10 by 1 pp; keep rectangular (notes 47b) |
| B3–B4 | **done** | device3 K=9: 69.9% ± 0.9 on true test 2 (notes 48) |
| B4 LOFO | **done** | Nov 25 24–28%; Mar 18 f1 55–64%; f2 reprints ~66–70% (notes 48b) |
| B5 | **done** | Best delta +0.6 pp (near-field), inside seed noise (notes 49–52) |
| B6–B8 | tooling only | Heatmaps / ensemble / ckpt-avg / mask-calib died on MPS; numbers not needed for the null |
| B9 | **not taken** | No cell beat B4 by more than seed spread *and* ≥9 pp; test 1 stays sealed |

**Test 1 discipline:** it has been sealed since day one and every choice in this repository was
made on test 2. A second score needs a preregistered gain of at least 9 pp. That bar was not met.

## Repository layout

```
src/config.py      inherited constants, split definitions, pinned microphone subsets
src/geometry.py    microphone positions and steering vectors, per session
src/labels.py      per-frame azimuth/elevation/range/speed from the GPS logs
src/stft_cache.py  the only module that opens a WAV; one pass -> cache/
src/train.py       trains one configuration -> runs/<run_id>/{best.pt, config.json, metrics.json}
src/predict.py     runs a checkpoint once -> results/preds/, which every analysis reads instead
src/detector.py    per-channel spectral detector, trained and evaluated on its own
src/esc50.py       ESC-50 confusers, mixed into our own spectra (detector only, never the map)
utils/check_*.py   one gate script per build step, each prints GATE PASS or GATE FAIL
                   check_vs_paper.py reproduces the reference paper's own metrics
utils/summarize_grid.py   the headline tables, seed spreads and array comparison
utils/search_*.py  the microphone subset searches (already run, results/subset_search.npz)
utils/plot_report_figures.py  the report's four-panel figure, assembled from results/*.json
src/polar.py       the zenith-disc projection: resampled remap, native steering, targets, decoding
src/demand.py      DEMAND ambient, level-matched to ours, for a quotable false-alarm rate
utils/run_grid.py  batteries of runs, sequential and resumable: step5a, headline, descriptive
utils/detach.py    launch a battery so it survives the shell that started it
utils/significance.py     intervals and p-values for every claim, corrected as one family
utils/history_by_range.py where the temporal-context gain lands, by distance bin
utils/detector_history.py the detector at K=1/5/9: site AUROC against unseen-ambient false alarms
utils/demand_trained.py   leave-one-group-out training on DEMAND ambient
utils/compare_polar.py    the five map cells: rectangle, resampled disc, native disc, ± Tversky
utils/confusion.py        confusion matrices at 95% recall, presence heads and detector
utils/calibration.py      presence-head reliability diagram
utils/error_structure.py  direction error by elevation and the shape of the azimuth error
utils/persistence_detector.py  direction stability as a parameter-free detection score
utils/hybrid_decoder.py   per-frame delay-and-sum / U-Net gate on peak contrast
```

Long runs go through the launcher, because macOS has no `setsid` and `nohup` alone does not
survive the parent's process group being signalled:

```bash
.venv/bin/python utils/detach.py logs/headline.log -- \
    .venv/bin/python -u utils/run_grid.py headline   # ~14 h, resumable, skips finished runs
tail -f logs/headline.log
```

## Five things that will bite you

1. **The only no-drone audio in the dataset is the tail of test_2's `flight_2`, and half of
   it is training data.** Split membership comes from each segment's declared `role`, never
   from the directory it sits in. `src/config.segments()` handles this; do not shortcut it.
2. **Never pre-cut the WAVs.** Segment bounds are `skip_seconds` / `skip_seconds_end` in
   each session's `alignment_params.json`, and the four recorders were started seconds
   apart, so each has its own `corrections_samples` offset.
3. **The array was recalibrated between sessions.** Microphone positions always come from
   that session's own JSON. `Oct_11_2024` differs from the others.
4. **`*_part1.wav` is capped at 2 GiB = 1864.1 s.** That is why `Mar_18_2025/flight_1` has
   36,140 label rows but only 17,685 usable frames — the paper used `part1` only, and we
   match it. Not a bug.
5. **Microphone subsets are pinned by channel list in `config.MIC_SUBSETS`.** The array has
   3-fold symmetry, so re-running the search can return a mirror image that scores
   identically. Report which channels were kept for every number.
6. **An AUROC of exactly 0.5000 is a bug, not a result.** It means every score is tied. We hit
   it twice: `sigmoid` saturating to 1.0 in float32, and the presence head's ReLU dying at
   lr 5e-3 in one seed while localization trained on happily. The epoch log now prints the
   presence-logit spread — if it reads 0.00, stop and fix rather than interpret.

## Conventions worth knowing before you write code

- A frame is 100 ms (4800 samples at 48 kHz), non-overlapping. GPS logs are 10 Hz, so
  **row k of the log is frame k of the segment** — that identity is the entire alignment
  scheme, and every gate script re-checks it.
- Energy maps are `(elevation 23, azimuth 91)`: elevation in rows, azimuth in columns.
  Azimuth is circular, so convolutions use `padding_mode='circular'` on that axis.
- The cache holds **381 native 10 Hz bins over 200–4000 Hz**, while the model consumes
  **16 averaged 125 Hz sub-bands over 200–2200 Hz**. Caching wide is what keeps the
  microphone subset, the direction grid, the map layout and the bin count as runtime flags
  instead of another pass over 24 GB of audio.
