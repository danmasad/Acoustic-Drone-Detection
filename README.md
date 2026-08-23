# Acoustic-Drone-Detection

Hostile drones are a growing security threat, including on Israel's northern front. Radio
detectors can fail when links are weak, jammed, encrypted, or missing. Acoustic detection is
a useful backup for early warning from sound alone.

Deep learning course project, 2026b. Dan and Shai.

## What we are actually claiming

We detect a drone and estimate its direction from a microphone array, and we ask one
question the reference paper left open:

> **Can temporal context let a cheap 6-microphone array match a 24-microphone one?**

**Measured answer, from 18 runs:** the cheap array matches all 24 already, with no temporal
context needed — 69.3% ± 0.7 against 70.5% ± 1.5 of frames within 10°, a gap inside the ±1.2 pp
seed spread. Temporal context adds +1.8 pp on the cheap array and nothing elsewhere. The
surprise is the third array: the subset our own search called optimal is 18 points *worse* than
a single off-the-shelf recorder. Thirteen arrays later, the reason is not the hardware but the
search — the beampattern criterion predicts accuracy at −0.80 across every array it did not
choose, and fails only on the winner it selected out of 106,183 candidates. Design notes
sections 21 and 22.

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
| [docs/report.md](docs/report.md) | **The final report, first draft.** Every number in it is measured; the gaps are marked `[TO FILL — …]` and listed under "What is left" below. Four pages of the five allowed |
| [docs/proposal_uav_acoustic_seld.md](docs/proposal_uav_acoustic_seld.md) | The submitted proposal (PDF built with `utils/build_pdf.sh`) |
| [docs/design_notes.md](docs/design_notes.md) | Everything measured, decided, or deliberately deferred. The reference document — 23 sections, including the microphone-subset search, the full experiment grid, and the result-by-result reading of the headline runs |
| `~/.cursor/plans/acoustic_uav_pipeline_013f8600.plan.md` | The executable build plan with a gate per step |

If you read one thing, read the report. If you read two, add design notes sections 21 and 22:
what the 18 headline runs say, and why the array our own search called optimal came last.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
utils/download_dataset.sh          # 24 GiB, ~1 h, resumable, safe to re-run
.venv/bin/python -m src.stft_cache # ~4 min, writes cache/ (3.4 GiB)
```

The dataset is the Western Sydney University UAV Acoustic Localization Dataset (CC BY-SA
4.0, no registration). `data/`, `cache/`, `logs/` and the 300 MB of checkpoints in `runs/`
are git-ignored, but **`runs/*/metrics.json` and `config.json` are committed**, so every
table in the report can be re-derived from a fresh clone with no audio and no compute:

```bash
.venv/bin/python utils/summarize_grid.py     # headline grid, order control, array comparison
```

`figures/` and `results/` are tracked for the same reason.

**Run anything that touches the GPU outside the agent sandbox.** Sandboxed shells report a
bogus "MPS requires macOS 14.0+" error.

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
| 6b. Other ablations | | K sweep, polar grid, bin count, speed head |

## What is left

In the order that matters, matching the `[TO FILL — …]` markers in the report:

| # | Task | Cost | Why it matters |
|---|---|---|---|
| 1 | **Unseal test 1.** Score the baseline, `device3` and `all24` at K=1 and K=5, and the detector, on the held-out site | ~20 min, no training | The whole report currently reports model-selection numbers. Nothing is quotable as generalization until this runs, and it runs **once** |
| 2 | **Division of work** paragraph | — | Explicitly required by the course instructions |
| 3 | **ESC-50 confuser rejection.** `src/esc50.py` is written, the data is downloaded, the spectral cache is not built | ~30 min | Turns "detection is data-limited" from an excuse into a measurement: which sounds actually read as a drone |
| 4 | Secondary ablations: polar map, bin count, K of 3 and 9, speed head | ~1 h each | Optional depth. Cut from the bottom if time runs out |
| 5 | Slides, 15 minutes | — | Graded separately |

**Test 1 discipline:** it has been sealed since day one and every choice in this repository was
made on test 2. Run it last, run it once, and report whatever it says.

## Repository layout

```
src/config.py      inherited constants, split definitions, pinned microphone subsets
src/geometry.py    microphone positions and steering vectors, per session
src/labels.py      per-frame azimuth/elevation/range/speed from the GPS logs
src/stft_cache.py  the only module that opens a WAV; one pass -> cache/
src/detector.py    per-channel spectral detector, trained and evaluated on its own
src/esc50.py       ESC-50 confusers, mixed into our own spectra (detector only, never the map)
utils/check_*.py   one gate script per build step, each prints GATE PASS or GATE FAIL
                   check_vs_paper.py reproduces the reference paper's own metrics
utils/summarize_grid.py   the headline tables, seed spreads and array comparison
utils/search_*.py  the microphone subset searches (already run, results/subset_search.npz)
utils/plot_*.py    the three report figures in figures/
utils/run_grid.py  batteries of runs, sequential and resumable: step5a, headline, descriptive
utils/detach.py    launch a battery so it survives the shell that started it
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
