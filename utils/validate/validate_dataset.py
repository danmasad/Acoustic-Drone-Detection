"""Feasibility check for the WSU UAV Acoustic Localization Dataset.

Streams a short slice of the 4x6-channel WAVs over HTTP range requests, runs
frequency-domain delay-and-sum beamforming on the paper's 91x23 az/el grid, and
compares the peak direction against the GPS-derived ground truth.
"""
import json, time, urllib.request
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pyproj import Transformer

BASE = ("https://rds.westernsydney.edu.au/Centres/ICNS/2026/Jahir_Rodriguez/"
        "dataset/test_1/Oct_11_2024/flight_5")
SR, CH, C = 48000, 6, 343.0
CHUNK = 4800                      # 100 ms frames, as in create_dataset.py
N_FRAMES = 150                    # 15 s
DATA_OFFSET = 44                  # PCM header size verified from the file
BPS = CH * 4                      # int32 samples

AZ = np.arange(-180, 181, 4)
EL = np.arange(0, 91, 4)
LOW, HIGH = 200.0, 2200.0         # paper's model input band


def fetch_range(url, first, last):
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    return urllib.request.urlopen(req, timeout=180).read()


def mic_positions(cfg):
    a, a2, h, r = cfg["a_"], cfg["a2_"], cfg["h_"], cfg["r_"]
    pos = []
    for ri in range(6):
        for ai in range(3):
            pos.append([r[ri] * np.cos(np.radians(a[ai])),
                        r[ri] * np.sin(np.radians(a[ai])), h[ri]])
    for ai in range(6):
        pos.append([r[2] * np.cos(np.radians(a2[ai])),
                    r[2] * np.sin(np.radians(a2[ai])), h[2]])
    return np.array(pos)


def steering(pos, freqs):
    az_g, el_g = np.meshgrid(np.radians(AZ), np.radians(EL), indexing="ij")
    d = np.stack([np.cos(el_g) * np.cos(az_g),
                  np.cos(el_g) * np.sin(az_g),
                  np.sin(el_g)], axis=-1).reshape(-1, 3)   # (D,3)
    tau = d @ pos.T / C                                    # (D,M)
    # (F,D,M) phase ramp
    return np.exp(-2j * np.pi * freqs[:, None, None] * tau[None]).astype(np.complex64)


def main():
    ap = json.load(open("alignment_params.json"))
    pos = mic_positions(ap["mic_array"])
    skip_n = int(ap["skip_seconds"] * SR)
    corr = ap["corrections_samples"]

    # --- stream the audio slice -------------------------------------------
    t0 = time.time()
    devs = []
    for i, name in enumerate(ap["wavs_nosync"]):
        start = skip_n + corr[i]
        first = DATA_OFFSET + start * BPS
        last = first + N_FRAMES * CHUNK * BPS - 1
        raw = fetch_range(f"{BASE}/{name}", first, last)
        devs.append(np.frombuffer(raw, dtype="<i4").reshape(-1, CH))
    audio = np.hstack(devs).astype(np.float64)             # (N, 24)
    dl_mb = audio.nbytes / 4 / 2**20
    print(f"streamed {dl_mb:.1f} MB -> audio {audio.shape} in {time.time()-t0:.1f}s")

    # --- GPS ground truth (mirrors create_dataset.py) ---------------------
    ref = pd.read_csv("ref5.csv", low_memory=False)
    fl = pd.read_csv("flight5.csv", low_memory=False)
    cols = ["latitude", "longitude", "altitude_above_seaLevel(feet)",
            "time(millisecond)", " xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"]
    fl = fl.iloc[ap["start_index_csv"]:].reset_index(drop=True)[cols].dropna()
    fl["altitude_above_seaLevel(feet)"] *= 0.3048
    alt0 = ref["altitude_above_seaLevel(feet)"].iloc[0] * 0.3048
    tr = Transformer.from_crs("epsg:4326", "epsg:32756", always_xy=True)
    rx, ry = tr.transform(ref["longitude"].dropna().astype(float).mean(),
                          ref["latitude"].dropna().astype(float).mean())
    fl["X_m"], fl["Y_m"] = tr.transform(fl["longitude"].values, fl["latitude"].values)

    def azim(x, y):
        return np.degrees(np.arctan2(y - ry, x - rx))

    az_off = ap["initial_azimuth_deg"] - azim(fl.iloc[0]["X_m"], fl.iloc[0]["Y_m"])

    # --- frequency-domain DAS --------------------------------------------
    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    band = (freqs >= LOW) & (freqs <= HIGH)
    W = steering(pos, freqs[band])
    print(f"steering tensor {W.shape} = {W.nbytes/2**20:.0f} MB, "
          f"{band.sum()} bins, {len(AZ)}x{len(EL)}={len(AZ)*len(EL)} directions")

    b, a = butter(5, [LOW / (SR / 2), 8000 / (SR / 2)], btype="band")
    rows, t_bf = [], 0.0
    for k in range(min(N_FRAMES, len(fl))):
        blk = filtfilt(b, a, audio[k * CHUNK:(k + 1) * CHUNK], axis=0)
        t1 = time.time()
        X = np.fft.rfft(blk, axis=0)[band].astype(np.complex64)      # (F,M)
        Y = np.einsum("fdm,fm->fd", W, X, optimize=True)             # (F,D)
        energy = (np.abs(Y) ** 2).sum(0).reshape(len(AZ), len(EL))
        t_bf += time.time() - t1

        i, j = np.unravel_index(energy.argmax(), energy.shape)
        row = fl.iloc[k]
        gt_az = -(((azim(row["X_m"], row["Y_m"]) + az_off + 180) % 360) - 180)
        horiz = np.hypot(row["X_m"] - rx, row["Y_m"] - ry)
        rel_alt = row["altitude_above_seaLevel(feet)"] - alt0
        gt_el = np.degrees(np.arctan2(rel_alt, horiz))
        dist = np.hypot(horiz, rel_alt)
        rows.append((abs(((AZ[i] - gt_az + 180) % 360) - 180), abs(EL[j] - gt_el), dist))

    e = np.array(rows)
    print(f"\nframes={len(e)}  beamforming {t_bf/len(e)*1000:.0f} ms/frame "
          f"({len(e)/t_bf:.1f} fps, single CPU core)")
    print(f"drone distance: {e[:,2].min():.0f}-{e[:,2].max():.0f} m")
    print(f"azimuth error   median {np.median(e[:,0]):.1f} deg   mean {e[:,0].mean():.1f}")
    print(f"elevation error median {np.median(e[:,1]):.1f} deg   mean {e[:,1].mean():.1f}")
    print(f"within 10 deg azimuth: {(e[:,0]<=10).mean()*100:.0f}% of frames")
    fmap = len(AZ) * len(EL) * band.sum() * 2 / 2**20
    print(f"\nfeature map per frame: {fmap:.2f} MB float32 (energy-only, 16 bins: "
          f"{len(AZ)*len(EL)*16*4/2**20:.2f} MB)")


if __name__ == "__main__":
    main()
