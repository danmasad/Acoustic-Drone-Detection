# Acoustic-Drone-Detection

Direction-of-arrival estimation and presence detection of a drone from a 24-microphone array (four 6-channel recorders, 48 kHz). Each 100 ms frame is band-passed, FFT'd, delay-and-sum beamformed onto a 91 x 23 azimuth/elevation grid (4 degree spacing), and fed to a U-Net that outputs a direction mask and a presence score. A kinematic filter on the decoded directions removes sidelobe hops. Dataset, splits and metrics follow Rodríguez et al. (2026).

```
src/                  library: config, geometry, beamforming, STFT cache, data, models, training, prediction
utils/setup/          dataset downloads, process detaching
utils/validate/       sanity checks on geometry, labels and the cache
utils/search/         microphone-subset search
utils/train/          batteries of training runs
utils/analysis/       tables from saved per-frame predictions
utils/plots/          figures
data/                 audio, GPS logs and alignment JSON (not in git)
cache/                per-segment STFT cache (not in git)
runs/<experiment>/    one directory per trained run: config.json, best.pt, metrics.json; runs/_archive/ holds superseded runs no result uses
results/preds/        per-frame prediction dumps, results/tables/ summary JSON
docs/                 report, design notes, proposal
```

Full method and measurements: [docs/report.md](docs/report.md), [docs/design_notes.md](docs/design_notes.md).

## Installation

Python 3.12. Pinned dependencies are in `requirements.txt` (CPU wheels for torch; on macOS the same wheel includes the MPS backend).

### macOS

```bash
git clone <repo-url> Acoustic-Drone-Detection && cd Acoustic-Drone-Detection
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

All commands below are written as `.venv/bin/python ...`. Device selection is automatic: MPS when available, otherwise CPU.

### Windows

Use PowerShell. The Python scripts run natively; only the two download shell scripts need a bash (Git Bash or WSL).

```powershell
git clone <repo-url> Acoustic-Drone-Detection; cd Acoustic-Drone-Detection
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Replace `.venv/bin/python` with `.venv\Scripts\python` in every command. For an NVIDIA GPU install the CUDA build of torch instead of the CPU wheel before the rest of the requirements (see pytorch.org for the index URL matching your driver), then pass `--device cuda`.

Checkpoints shipped in `runs/` record `"device": "mps"` in their `config.json`, and `src.predict` uses that value as given. Off a Mac, always pass `--device cpu` (or `cuda`) to `src.predict`, `src.train`, and `src.baseline`.

## Data

### Download

The audio is not in the repository. `utils/setup/dataset_manifest.tsv` lists every file (URL path, destination, byte size) and `utils/setup/download_dataset.sh` fetches them, two at a time, resumable, skipping files already complete.

```bash
bash utils/setup/download_dataset.sh            # ~24 GB into data/
```

The script uses the BSD `stat -f%z`; under Git Bash or WSL change the three occurrences to `stat -c%s`. Any other downloader works as long as the layout below is reproduced.

Optional, only for the per-channel presence detector (`src.detector`): `utils/setup/download_demand.sh` (DEMAND ambient scenes, ~520 MB) and `utils/setup/download_esc50.sh` (ESC-50 confusers, ~600 MB).

### Layout

`src/config.py: SESSION_DIRS` names the four sessions; nothing else is scanned.

```
data/training/Nov_25_2024/flight_11/
data/training/Mar_18_2025/flight_1/
data/test_2/Mar_18_2025/flight_2/
data/test_1/Oct_11_2024/flight_5/
```

Each session directory holds:

- four WAVs, one per recorder, 6 channels, 48 kHz, int32 (`wavs_nosync` in the JSON, in recorder order 1..4 so channel `6*(d-1)+c` is recorder `d`, mic `c`);
- the drone's GPS log (`flight_csv`, Airdata export at 10 Hz) and the stationary reference log (`ref_csv`, under `data/<split>/<session>/ref/`);
- `alignment_params.json`: `session`, `flight_id`, `role` (`training` / `test_1` / `test_2`), `sample_rate_hz`, `corrections_samples` (per-recorder sample offset so all 24 channels are time-aligned), `mic_array` (geometry), `wavs_nosync`, and either top-level segment fields or a `segments` list, each with `skip_seconds`, optional `skip_seconds_end`, `flight_csv` (null for a no-drone segment), `start_index_csv`, `initial_azimuth_deg`.

Split membership comes from each segment's `role`, not from the directory. The minutes of pre-takeoff audio in every WAV are mined automatically as an extra no-drone segment of the same split (`config._pre_takeoff_spec`).

### Build the STFT cache

One pass over the audio writes, per segment, `cache/<session>_f<flight>_<name>/{stft.npy, meta.npz}`: complex64 spectra `(frames, 24, bins)` for 200-4000 Hz plus the per-frame labels (azimuth, elevation, range, speed, has_drone). About 4 GB total; everything downstream reads this cache, never the WAVs.

```bash
.venv/bin/python -m src.stft_cache                 # all segments, skips those present
.venv/bin/python -m src.stft_cache --only Mar_18   # substring filter on segment id
.venv/bin/python -m src.stft_cache --overwrite
```

Checks, in the order they should pass:

```bash
.venv/bin/python utils/validate/check_geometry_labels.py   # GPS labels vs beamformed peaks
.venv/bin/python utils/validate/check_stft_cache.py        # cache == recomputed from WAV, prints GATE PASS
```

## Training from scratch

`src.train` trains one U-Net on the `train` split, early-stops on `test_2` hit rate at 10 degrees, and writes `runs/<experiment>/<run_id>/{config.json, best.pt, metrics.json}`. The run id is built by `config.run_id()` as `<subset>_K<K>_w<base>_b<subbands>_<loss>[_polar|_polarn]_s<seed>[_<tag>]`; `config.experiment_of()` chooses the experiment folder from the id.

```bash
.venv/bin/python -m src.train --subset device3 --K 1 --seed 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--subset` | `device3` | microphone subset, a key of `config.MIC_SUBSETS` (`all24`, `device3`, `best6`, `device1`, ...) |
| `--K` | `1` | number of consecutive frames stacked as input channels |
| `--subbands` | `16` | frequency sub-bands per frame between 200 and 2200 Hz |
| `--base`, `--depth` | `32`, `3` | U-Net width and depth |
| `--loss` | `wce` | `wce` or `tversky` |
| `--polar`, `--disc` | off, `resample` | train on the polar disc instead of the rectangular grid; `--disc native` steers the beamformer on the disc |
| `--lr`, `--batch` | `5e-3`, `32` | optimizer |
| `--max-epochs`, `--patience`, `--min-epochs` | `60`, `10`, `15` | early stopping |
| `--seed` | `0` | seed |
| `--device` | `auto` | `mps`, `cuda`, or `cpu` |
| `--tag` | `""` | suffix appended to the run id |

`src.train` overwrites an existing run directory of the same id. `utils/train/run_grid.py <battery> [<battery> ...]` runs a named list of cells (`headline` is the 3 subsets x 2 K x 3 seeds grid; see `BATTERIES` in the file) and skips any cell whose `metrics.json` exists. For a multi-hour battery:

```bash
.venv/bin/python utils/setup/detach.py logs/headline.log -- .venv/bin/python utils/train/run_grid.py headline
```

Delay-and-sum baseline (no training) and the separate presence detector:

```bash
.venv/bin/python -m src.baseline --subsets all24 device3 best6 --splits test_2
.venv/bin/python -m src.detector --K 1 --seed 0        # needs data/demand and data/esc50
```

After training, dump per-frame predictions and confirm they reproduce `metrics.json`:

```bash
.venv/bin/python -m src.predict --split test_2 --runs device3_K1_w32_b16_wce_s0   # prints GATE PASS
.venv/bin/python utils/analysis/summarize_grid.py
```

The analysis scripts under `utils/analysis/` read only `results/preds/` and `runs/*/*/metrics.json`, so the tables rebuild without audio or a GPU.

## Running a trained model on a new recording

A checkpoint is a bare state dict; the architecture, microphone channels and input normalization live in the `config.json` beside it. Always load through `src.predict.cnn_predictions` (or `src.detector.load_run` for the detector). Inference needs the same array: four recorders, same channel order, and a `mic_array` geometry in the session's `alignment_params.json` (the steering tensor is rebuilt from it per session, so a re-measured geometry is fine).

1. Put the recording in its own session directory with the four WAVs and an `alignment_params.json` as described under Layout. Set `role` to the split you want the dump filed under (`test_1`, `test_2`, or `training`). With a GPS log, fill `flight_csv`, `ref_csv`, `start_index_csv` and `initial_azimuth_deg` and the dump will carry ground truth; without one set `"flight_csv": null` and give `skip_seconds` / `skip_seconds_end`, and every frame is labelled no-drone (predictions are still produced).

2. Register it in `src/config.py`:

   ```python
   SESSION_DIRS = [
       ...,
       DATA / "test_1/<Session>/flight_<n>",
   ]
   ```

3. Cache and predict with an existing run:

   ```bash
   .venv/bin/python -m src.stft_cache --only <Session>
   .venv/bin/python -m src.predict --split test_1 --runs device3_K1_w32_b16_wce_s0 --device cpu --overwrite
   ```

   `--overwrite` is required because `results/preds/<experiment>/<run_id>_<split>_causal.npz` already exists for the shipped sessions; the new dump covers every segment of that split, old and new. `GATE FAIL` is expected here, since the dump no longer matches the `metrics.json` computed on the original split.

4. Read the predictions:

   ```python
   from src.predict import load
   d = load("device3_K1_w32_b16_wce_s0_test_1_causal")
   seg_names = d["seg_ids"]                 # index into this with d["seg"]
   new = d["seg"] == list(seg_names).index("<Session>_f<n>_drone")
   d["score"][new]                          # presence logit per 100 ms frame
   d["pred_az"][new], d["pred_el"][new]     # decoded direction, degrees (centroid decoder)
   d["argmax_az"][new], d["argmax_el"][new] # argmax decoder
   d["frame"][new]                          # frame index inside the segment
   ```

   `score` is a logit: threshold it against the operating point you need (`utils/analysis/detection_range.py` shows how a threshold is set from no-drone frames). `utils/analysis/score_kinematic_filter.py: apply_filter` is the post-filter that removes lobe hops from `pred_az`/`pred_el`.

To use a run trained on a different subset (`all24`, `best6`), pass its run id; the channels come from its `config.json`.
