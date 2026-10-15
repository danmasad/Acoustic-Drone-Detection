# Acoustic-Drone-Detection

Direction-of-arrival estimation and presence detection of a drone from a 24-microphone array (four 6-channel recorders, 48 kHz). Each 100 ms frame is band-passed, FFT'd, delay-and-sum beamformed onto a 91 x 23 azimuth/elevation grid (4 degree spacing), and fed to a U-Net that outputs a direction mask and a presence score. A kinematic filter on the decoded directions removes sidelobe hops. A separate one-microphone 1-D CNN detects presence on its own. Dataset, splits and metrics follow Rodríguez et al. (2026).

Full method and measurements: [docs/report.pdf](docs/report.pdf). Slides: [docs/presentation.pptx](docs/presentation.pptx).

This repository holds what is needed to set up the environment, download the data, retrain the models and run data through them.

```
src/                  library: config, geometry, beamforming, STFT cache, data, models, training, prediction,
                      kinematic filter, one-microphone detector (+ DEMAND / ESC-50 loaders)
utils/setup/          dataset downloads, process detaching
utils/train/          batteries of U-Net runs, detector leave-one-scene-out and DEMAND experiments
utils/validate/       check_stft_cache.py, the gate that the cache matches the audio
runs/<experiment>/    every trained run: config.json, metrics.json, and best.pt for the runs the report quotes
results/preds/        per-frame prediction dumps of the shipped runs; results/tables/ summary JSON
docs/                 report.pdf, presentation.pptx
data/                 audio, GPS logs and alignment JSON (downloaded, not in git)
cache/                per-segment STFT cache (built, not in git)
```

## 1. Setup

### Environment

Python 3.12. Pinned dependencies are in `requirements.txt` (CPU wheels for torch; on macOS the same wheel includes the MPS backend).

macOS / Linux:

```bash
git clone <repo-url> Acoustic-Drone-Detection && cd Acoustic-Drone-Detection
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Windows (PowerShell; the ESC-50 and DEMAND download scripts need Git Bash or WSL):

```powershell
git clone <repo-url> Acoustic-Drone-Detection; cd Acoustic-Drone-Detection
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

All commands below are written as `.venv/bin/python ...`; on Windows use `.venv\Scripts\python`. Device selection is automatic: MPS when available, otherwise CPU. For an NVIDIA GPU install the CUDA build of torch first (see pytorch.org) and pass `--device cuda`.

The shipped checkpoints record `"device": "mps"` in their `config.json`, and `src.predict` uses that value as given. Off a Mac, always pass `--device cpu` (or `cuda`) to `src.predict`, `src.train` and `src.baseline`.

### Download the dataset

The data is the UAV Acoustic Localization Dataset (Rodríguez et al., Western Sydney University, [doi:10.26183/07nm-4p35](https://doi.org/10.26183/07nm-4p35)), hosted at:

<https://rds.westernsydney.edu.au/Institutes/MARCS/ICNS/2026/Jahir_Rodriguez/>

Only its `dataset/` folder is needed, and within it only the files in `utils/setup/dataset_manifest.tsv` (URL path, destination, byte size): the four sessions' WAVs, GPS logs and `alignment_params.json`, not the reference video. The script fetches them two at a time, resumes partial files, and skips files already complete. Safe to re-run.

```bash
.venv/bin/python utils/setup/download_dataset.py     # ~24 GB into data/
```

If the script fails (the host has changed address before, and rejects non-browser clients with HTTP 403), download the same files from the link above in a browser and place them under `data/` with the paths listed in the manifest, which mirror the `dataset/` folder.

Only for the one-microphone detector (`src.detector`):

```bash
bash utils/setup/download_esc50.sh              # ESC-50 confuser sounds, ~600 MB into data/esc50
bash utils/setup/download_demand.sh             # DEMAND ambient scenes: ~5.4 GB downloaded, ~520 MB kept in data/demand
```

DEMAND comes from Zenodo, which can be slow (hours at a few hundred KB/s); both scripts resume where they stopped.

### Data layout

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

One pass over the audio writes, per segment, `cache/<session>_f<flight>_<name>/{stft.npy, meta.npz}`: complex64 spectra `(frames, 24, bins)` for 200-4000 Hz plus the per-frame labels (azimuth, elevation, range, speed, has_drone). About 4 GB; everything downstream reads this cache, never the WAVs.

```bash
.venv/bin/python -m src.stft_cache                 # all segments, skips those present
.venv/bin/python -m src.stft_cache --only Mar_18   # substring filter on segment id
.venv/bin/python -m src.stft_cache --overwrite
.venv/bin/python utils/validate/check_stft_cache.py   # cache == recomputed from WAV, prints GATE PASS
```

## 2. Train

### One U-Net

`src.train` trains one U-Net on the `train` split, early-stops on `test_2` hit rate at 10 degrees, and writes `runs/<experiment>/<run_id>/{config.json, best.pt, metrics.json}`. The run id is `<subset>_K<K>_w<base>_b<subbands>_<loss>[_polar|_polarn]_s<seed>[_<tag>]` (`config.run_id()`); `config.experiment_of()` picks the experiment folder from it.

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

`src.train` overwrites an existing run directory of the same id.

### The report's U-Net runs

`utils/train/run_grid.py <battery> [...]` runs a named list of cells (see `BATTERIES` in the file) and skips any cell whose `metrics.json` already exists, so delete or move a run directory to retrain it.

| Battery | What the report uses it for |
|---|---|
| `headline` | all24 / device3 / best6 x K = 1, 5 x three seeds |
| `k9`, `all24k9` | K = 9 on device3 and all24, three seeds |
| `tversky` | Tversky loss against class-weighted cross-entropy |
| `polar`, `polarnative`, `polarnative_tversky`, `polarnative_k9` | polar disc against the rectangular map |
| `coherence`, `bins` | other microphone subsets, 8 and 32 sub-bands |

Batteries take hours; detach so a closed terminal does not stop them:

```bash
.venv/bin/python utils/setup/detach.py logs/headline.log -- .venv/bin/python -u utils/train/run_grid.py headline k9 all24k9
```

### Delay-and-sum baseline (no training)

```bash
.venv/bin/python -m src.baseline --subsets all24 device3 best6 --splits test_2
```

### One-microphone detector

Needs `data/esc50` and `data/demand` (see Setup). The four full-data detectors the report quotes:

```bash
.venv/bin/python -m src.detector --tag full                                # detector_K1_w32_s0_full
.venv/bin/python -m src.detector --tag full --log-bins 64                  # detector_K1_w32_log64_s0_full
.venv/bin/python -m src.detector --tag full --confusers                    # detector_K1_w32_conf_s0_full, ESC-50 mixed in
.venv/bin/python -m src.detector --tag full --confusers --log-bins 64      # detector_K1_w32_conf_log64_s0_full
```

Leave-one-scene-out (each of the three recordings held out in turn) and DEMAND hold-one-group-out:

```bash
.venv/bin/python utils/train/detector_loso.py [--confusers] [--log-bins 64]
.venv/bin/python utils/train/demand_trained.py
.venv/bin/python utils/train/demand_false_alarms.py detector_K1_w32_conf_s0_full   # false alarms per hour on DEMAND
```

`demand_trained.py` reuses any fold whose `runs/detector/detector_K1_w32_amb_s0_full_no<group>/metrics.json` and `results/tables/detector/demand_false_alarms_<run>.json` already exist, so with the shipped files it only rebuilds the summary; delete those to retrain.

## 3. Run data through a trained model

### Shipped models

`runs/` holds every run's `config.json` and `metrics.json`. Weights (`best.pt`) are shipped for the 24 grid U-Nets (`all24`, `device3` at K = 1, 5, 9 and `best6` at K = 1, 5, three seeds each) and the four full-data detectors. Quote the three seeds together: seed 0 is device3's best and all24's worst, so one checkpoint per cell would make device3 look better than all24, while the claim is that the two are within the seed noise.

A checkpoint is a bare state dict; the architecture, microphone channels and input normalization live in the `config.json` beside it. Always load through `src.predict.cnn_predictions` (or `src.detector.load_run` for the detector).

### Reproduce a run's metrics on the shipped splits

With the cache built:

```bash
.venv/bin/python -m src.predict --split test_2 --runs device3_K1_w32_b16_wce_s0 --device cpu --overwrite   # prints GATE PASS
```

The dump is written to `results/preds/<experiment>/<run_id>_<split>_causal.npz` and checked against the run's `metrics.json`. Without `--overwrite` the shipped dump is read back instead of recomputed.

### A new recording

Inference needs the same array: four recorders, same channel order, and a `mic_array` geometry in the session's `alignment_params.json` (steering is rebuilt from it per session, so a re-measured geometry is fine).

1. Put the recording in its own session directory with the four WAVs and an `alignment_params.json` as described under Data layout. Set `role` to the split you want the dump filed under (`test_1`, `test_2`, or `training`). With a GPS log, fill `flight_csv`, `ref_csv`, `start_index_csv` and `initial_azimuth_deg` and the dump carries ground truth; without one set `"flight_csv": null` and give `skip_seconds` / `skip_seconds_end`, and every frame is labelled no-drone (predictions are still produced).

2. Register it in `src/config.py`:

   ```python
   SESSION_DIRS = [
       ...,
       DATA / "test_1/<Session>/flight_<n>",
   ]
   ```

3. Cache and predict:

   ```bash
   .venv/bin/python -m src.stft_cache --only <Session>
   .venv/bin/python -m src.predict --split test_1 --runs device3_K1_w32_b16_wce_s0 --device cpu --overwrite
   ```

   The new dump covers every segment of that split, old and new, so `GATE FAIL` is expected: it no longer matches the `metrics.json` computed on the original split.

4. Read the predictions and apply the kinematic filter:

   ```python
   import numpy as np
   from src.predict import load
   from src.kinematic import apply_filter

   d = load("device3_K1_w32_b16_wce_s0_test_1_causal")
   new = d["seg"] == list(d["seg_ids"]).index("<Session>_f<n>_drone")
   order = np.argsort(d["frame"][new])
   score = d["score"][new][order]                      # presence logit per 100 ms frame
   az, el = d["pred_az"][new][order], d["pred_el"][new][order]   # centroid decoder, degrees
   f_az, f_el = apply_filter(az, el)                   # sidelobe hops removed
   ```

   `score` is a logit: set a threshold from no-drone frames at the operating point you need. `argmax_az` / `argmax_el` hold the brightest-cell decoder. Run `apply_filter` once per segment, in frame order.

To use a run trained on a different subset (`all24`, `best6`), pass its run id; the channels come from its `config.json`.
