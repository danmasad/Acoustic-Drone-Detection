# Acoustic-Drone-Detection

Six microphones on one recorder match a 24-microphone array at naming a drone's direction, ten times a second. Extra frames of history help only where microphones are scarce. A kinematic filter after the network removes lobe hops. It does not produce the far-range accuracy in Rodríguez et al. (2026), and it does not fix the near field.

Deep learning course project, 2026b. Dan Masad and Shai Livshits.

## Documents

| File | What it is |
|---|---|
| [docs/report.md](docs/report.md) | The report. [docs/report.pdf](docs/report.pdf) is the same text |
| [docs/presentation.pptx](docs/presentation.pptx) | The talk, built from the report |
| [docs/design_notes.md](docs/design_notes.md) | The lab notebook: what was measured, and in what order |
| [docs/proposal_uav_acoustic_seld.md](docs/proposal_uav_acoustic_seld.md) | The proposal |

## Presentation

The deck is [docs/presentation.pptx](docs/presentation.pptx). [figures/flight_hemisphere_mar18_f2.gif](figures/flight_hemisphere_mar18_f2.gif) is the flight on the title slide. [figures/angular_error_hemisphere.png](figures/angular_error_hemisphere.png) and [figures/angular_error_hemisphere_rotate.gif](figures/angular_error_hemisphere_rotate.gif) are a 10° cone on the same sky, so a hit rate has a picture.

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/build_slides.py
MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/plot_array_geometry.py --slides
MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/plot_flight_hemisphere.py
MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/plot_angular_error_hemisphere.py
MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/plot_angular_error_hemisphere.py --gif
```

`plot_array_geometry.py --slides` writes [figures/array_tested.png](figures/array_tested.png). Every panel is the same microphone map. The new site is that map after the microphones moved.

## Reproduce the numbers

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The per-frame dumps are in the repository, so the tables rebuild with no audio and no GPU. `score_kinematic_filter.py` prints two blocks. The first is the paper's metric, and it is the only block to set beside their figure. The second is ours: every drone frame, our distance bins, and the sideways miss.

```bash
.venv/bin/python utils/analysis/summarize_grid.py
.venv/bin/python utils/analysis/check_vs_paper.py
.venv/bin/python utils/analysis/score_kinematic_filter.py
.venv/bin/python utils/analysis/score_kinematic_filter.py --split test_1
.venv/bin/python utils/analysis/check_history_depth.py
.venv/bin/python utils/analysis/history_by_range.py
.venv/bin/python utils/analysis/significance.py
.venv/bin/python utils/analysis/compare_polar.py
.venv/bin/python utils/analysis/error_structure.py
.venv/bin/python utils/analysis/hybrid_decoder.py
```

### Dataset

The audio is not in the repository. `utils/setup/download_dataset.sh` fetches the four flights (about 24 GB, resumable). Lay them out as:

```
data/training/Nov_25_2024/flight_11/
data/training/Mar_18_2025/flight_1/
data/test_2/Mar_18_2025/flight_2/
data/test_1/Oct_11_2024/flight_5/
```

Each directory holds the four recorder WAVs, the GPS csv, and `alignment_params.json`. Then:

```bash
.venv/bin/python -m src.stft_cache
.venv/bin/python utils/validate/check_stft_cache.py
```

### Run a saved model

Checkpoints live in `runs/<experiment>/<run_id>/`. Experiments are `grid`, `loss`, `disc`, `arrays`, `bins`, `probes`, and `detector`. A dump is `results/preds/<experiment>/<run_id>_<split>_<order>.npz`. Summary tables are under `results/tables/`.

```bash
.venv/bin/python -m src.predict --split test_2 --runs device3_K1_w32_b16_wce_s0
```

That reprints `GATE PASS` when the dump matches the run's own `metrics.json`. Training a new cell:

```bash
.venv/bin/python -m src.train --subset device3 --K 5 --seed 0
.venv/bin/python utils/train/run_grid.py headline
```

`utils/train/run_grid.py` skips a cell whose `metrics.json` is already on disk. `utils/setup/detach.py` starts a long run in its own session.

## Layout

```
src/                  the library: geometry, beamforming, models, training, prediction
utils/setup/          downloads, detach, the PDF build
utils/validate/       checks that run before a model does
utils/search/         the microphone-subset search
utils/train/          batteries of training runs
utils/analysis/       numbers from saved predictions
utils/plots/          figures and the deck
runs/<experiment>/    one directory per trained run
results/preds/        per-frame dumps, mirrored by experiment
results/tables/       summary JSON, by the question it answers
results/search/       the microphone-search cache
figures/              pictures, including the ones on the slides
```

## Two footguns

`best.pt` is a bare state dict. The architecture and the input normalization live in the JSON next to it. Load through `predict.cnn_predictions` or `detector.load_run`.

Off a Mac, pass `device` explicitly. The saved configs say `mps`, and that value is used as given. On Linux or CUDA use `device="cpu"` or `--device cpu`.
