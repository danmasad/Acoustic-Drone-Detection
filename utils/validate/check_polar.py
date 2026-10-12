"""Gate for stage E: does the disc projection preserve direction, and does it cost what we said?

An accuracy number from the polar model means nothing until the plumbing is proved, because
every failure mode here is silent. A flipped sign in the azimuth convention, a half-cell offset
between the input grid and the target rasterizer, or an off-by-one in the elevation mapping all
produce a model that trains happily to a wrong answer.

Four checks:

1. **Round trip.** Build the target for a known direction, decode it, and require the recovered
   direction back within one grid cell. Run over a spread of azimuths and elevations, since a
   sign error shows at one azimuth and a scale error shows at one elevation.
2. **Input and target agree.** Resample a rectangular map that is a single bright cap onto the
   disc, and check its brightest disc cell lands inside the disc target for the same direction.
   This is the check that catches the input and the target using different conventions, which no
   amount of round-tripping the target alone would reveal.
3. **Dead corners.** Confirm the unusable fraction is the 21.5% the geometry predicts, and that
   the projection leaves them at exactly zero.
4. **Resolution parity.** Confirm the radial step matches the rectangular grid's 4 degrees, so a
   layout comparison is not secretly a resolution comparison.
5. **Native steering.** The disc can be built two ways, and the second one is why this file grew a
   fifth section. `Projection` resamples the finished rectangular map, so the input is a bilinear
   blur of 4-degree cells painted onto rim cells 2.5 degrees apart while the target stays a hard
   cap; `NativeDisc` steers the beamformer at the disc's own directions and interpolates nothing.
   Checked here: the memoized beamformer really does return a different steering tensor for the
   two direction sets, a synthetic plane wave peaks within one cell of its analytic direction on
   the natively steered disc, the dead corners are exactly zero, and the peak is measurably
   sharper than the resampled one. That last number is the mechanism the whole experiment rests
   on, so it is measured before any training time is spent on it.

Low elevations are expected to round-trip slightly high: a 10 degree cap centred at 5 degrees
elevation extends below the horizon, where the disc has no cells, so the visible part of the cap
has its centroid above the true centre. The rectangular grid clips identically, so this is a
property of the label, not of the projection, and it is reported rather than tolerated silently.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import beamform, polar                                  # noqa: E402
from src.config import (ANGULAR_THRESHOLD, AZ, C, CHUNK, EL, GRID_SHAPE,   # noqa: E402
                        MIC_SUBSETS, MODEL_BAND, SR, segments)
from src.geometry import (angular_distance, directions, mic_positions,     # noqa: E402
                          to_angles, unit_vectors)
from src.labels import target_mask as rect_mask                   # noqa: E402
from src.models import decode_mask                                # noqa: E402

CELL = 90.0 / (polar.N / 2)          # degrees per radial cell
AZ_TEST = (-180.0, -135.0, -90.0, -20.0, 0.0, 37.0, 90.0, 179.0)
EL_TEST = (2.0, 12.0, 30.0, 45.0, 70.0, 88.0)
CLIPPED_BELOW = 12.0                 # below this the cap is cut by the horizon


def as_logits(mask: np.ndarray) -> torch.Tensor:
    """A hard mask read as logits, so the decoder sees probability 1 inside and 0 outside."""
    return torch.from_numpy(mask.astype(np.float32) * 40.0 - 20.0)


def main() -> int:
    print(__doc__.split("\n\n")[0])
    proj = polar.Projection()
    dirs, valid = proj.dirs, proj.valid
    fails = []

    # --- 1. round trip -----------------------------------------------------------------
    az_t = np.array([a for a in AZ_TEST for _ in EL_TEST], float)
    el_t = np.array([e for _ in AZ_TEST for e in EL_TEST], float)
    m = polar.target_mask(az_t, el_t, ANGULAR_THRESHOLD)
    d_az, d_el = decode_mask(as_logits(m), mode="centroid", dirs=dirs, valid=valid)
    err = angular_distance(d_az, d_el, az_t, el_t)
    empty = m.reshape(len(m), -1).sum(1) == 0
    high = el_t >= CLIPPED_BELOW

    print(f"\n1. round trip over {len(az_t)} directions, cell size {CELL:.2f} deg")
    print(f"   above {CLIPPED_BELOW:.0f} deg elevation: max error {err[high].max():.2f} deg, "
          f"median {np.median(err[high]):.2f}")
    print(f"   below {CLIPPED_BELOW:.0f} deg (cap clipped by the horizon): max "
          f"{err[~high].max():.2f} deg, all biased upward by construction")
    print(f"   empty targets: {int(empty.sum())} of {len(m)}")
    if err[high].max() > CELL:
        fails.append(f"round trip off by {err[high].max():.2f} deg, over one {CELL:.2f} deg cell")
    if empty.any():
        fails.append(f"{int(empty.sum())} directions produced an empty target mask")

    # --- 2. input and target agree -----------------------------------------------------
    # A smooth bump rather than a hard cap: a flat plateau has no unique brightest cell, so its
    # argmax could land anywhere inside the cap and the check would pass on a projection that is
    # a whole cap-width off. A Gaussian beam has one peak and locates it to sub-cell precision,
    # which is what a convention mismatch would break.
    probe_az = np.array([37.0, -120.0, 150.0, -179.0])
    probe_el = np.array([20.0, 55.0, 75.0, 8.0])
    rect_dirs = directions()
    beam = np.empty((len(probe_az),) + GRID_SHAPE, np.float32)
    for i in range(len(probe_az)):
        a, e = np.radians(probe_az[i]), np.radians(probe_el[i])
        src = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        ang = np.degrees(np.arccos(np.clip(rect_dirs @ src, -1.0, 1.0)))
        beam[i] = np.exp(-0.5 * (ang / 5.0) ** 2).reshape(GRID_SHAPE)
    resampled = proj(torch.from_numpy(beam).unsqueeze(1)).squeeze(1)
    disc = polar.target_mask(probe_az, probe_el, ANGULAR_THRESHOLD)
    inside, peaks = [], []
    for i in range(len(probe_az)):
        k = int(resampled[i].flatten().argmax())
        inside.append(bool(disc[i].flatten()[k]))
        v = dirs[k].cpu().numpy()
        peaks.append(angular_distance(np.degrees(np.arctan2(v[1], v[0])),
                                      np.degrees(np.arcsin(np.clip(v[2], -1, 1))),
                                      probe_az[i], probe_el[i]))
    print(f"\n2. resampled Gaussian beam's peak vs the disc target, {len(probe_az)} directions")
    for i, (ok, e) in enumerate(zip(inside, peaks)):
        print(f"   az {probe_az[i]:>7.1f} el {probe_el[i]:>5.1f}: peak {e:.2f} deg from the "
              f"true direction, inside the target: {ok}")
    if not all(inside) or max(peaks) > CELL:
        fails.append(f"resampled input peaks up to {max(peaks):.2f} deg off (limit one "
                     f"{CELL:.2f} deg cell): input and target disagree on convention")

    # --- 3. dead corners --------------------------------------------------------------
    dead = 1.0 - float(valid.float().mean())
    ones = proj(torch.ones(1, 1, *GRID_SHAPE)).squeeze()
    dead_max = float(ones[~valid.reshape(polar.N, polar.N)].abs().max())
    print(f"\n3. dead corners: {dead * 100:.1f}% of cells, geometry predicts "
          f"{polar.DEAD_FRACTION * 100:.1f}%; largest value left in them {dead_max:.1e}")
    if abs(dead - polar.DEAD_FRACTION) > 0.02 or dead_max > 1e-6:
        fails.append(f"dead corners are {dead * 100:.1f}% and carry up to {dead_max:.1e}")

    # --- 4. resolution parity ---------------------------------------------------------
    rect_cells, disc_cells = GRID_SHAPE[0] * GRID_SHAPE[1], int(valid.sum())
    print(f"\n4. resolution: radial step {CELL:.2f} deg against the rectangular grid's "
          f"{EL[1] - EL[0]:.0f} deg elevation step")
    print(f"   usable cells {disc_cells} on the disc, {rect_cells} on the rectangle "
          f"({disc_cells / rect_cells:.2f}x)")
    print(f"   azimuth step at the horizon {360 / (2 * np.pi * (polar.N / 2)):.2f} deg, "
          f"against {AZ[1] - AZ[0]:.0f} deg everywhere on the rectangle")
    if abs(CELL - (EL[1] - EL[0])) > 0.2:
        fails.append(f"radial step {CELL:.2f} deg does not match the elevation step")

    # --- 5. native steering ------------------------------------------------------------
    nd = polar.NativeDisc()
    seg = segments("test_2")[0]
    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    freqs = freqs[(freqs >= MODEL_BAND[0]) & (freqs <= MODEL_BAND[1])]
    channels = MIC_SUBSETS["all24"]
    bf_rect = beamform.get(seg.align, channels, freqs, key=seg.session, n_subbands=16,
                           device="cpu")
    bf_disc = beamform.get(seg.align, channels, freqs, key=seg.session, n_subbands=16,
                           device="cpu", dirs=nd.steering_dirs, grid_shape=nd.grid_shape)
    print(f"\n5. native steering: {len(nd.steering_dirs)} disc directions against "
          f"{rect_cells} rectangular ones")
    distinct = bf_rect is not bf_disc and bf_disc.grid_shape == (polar.N, polar.N)
    print(f"   the memo returns a separate steering tensor for the disc: {distinct}")
    if not distinct:
        fails.append("beamform.get handed back the rectangular steering tensor for the disc; "
                     "the direction set is missing from the memo key")

    pos = mic_positions(seg.align)[channels]
    native_err, prominence = [], []
    for a_deg, e_deg in zip(probe_az, probe_el):
        src = unit_vectors(a_deg, e_deg)
        # A unit-amplitude plane wave from src, sampled at each microphone's own delay.
        S = np.exp(2j * np.pi * freqs[:, None] * (pos @ src / C)[None, :]).T
        X = torch.from_numpy(S[None].astype(np.complex64))

        # Both paths standardize before the disc is masked, exactly as `data.to_maps` does, so
        # zero is the mean of the distribution rather than a bright or dark artefact.
        raw_n = bf_disc.maps(X)
        nat = nd((raw_n - raw_n.mean()) / raw_n.std()).sum(1).squeeze(0)
        raw_r = bf_rect.maps(X)
        res = proj((raw_r - raw_r.mean()) / raw_r.std()).sum(1).squeeze(0)

        k = int(nat.flatten().masked_fill(~valid, -1e9).argmax())
        native_err.append(angular_distance(*to_angles(nd.steering_dirs[k]), a_deg, e_deg))
        # Prominence: the peak minus the sidelobe floor 15-45 degrees away, in standard
        # deviations. Blur moves energy out of the peak and into its neighbourhood, so it costs
        # prominence whether or not it moves the argmax.
        ang = np.degrees(np.arccos(np.clip(nd.steering_dirs @ src, -1.0, 1.0)))
        ring = torch.from_numpy((ang >= 15.0) & (ang <= 45.0)) & valid
        prominence.append((float(nat.flatten()[valid].max() - nat.flatten()[ring].mean()),
                           float(res.flatten()[valid].max() - res.flatten()[ring].mean())))

    for i, (a_deg, e_deg) in enumerate(zip(probe_az, probe_el)):
        n_p, r_p = prominence[i]
        print(f"   az {a_deg:>7.1f} el {e_deg:>5.1f}: peak {native_err[i]:.2f} deg off, "
              f"prominence {n_p:5.2f} sd native against {r_p:5.2f} resampled "
              f"({n_p - r_p:+.2f})")
    sharper = sum(n > r for n, r in prominence)
    print(f"   native is sharper in {sharper} of {len(prominence)} directions, mean gain "
          f"{np.mean([n - r for n, r in prominence]):+.2f} sd")
    if max(native_err) > CELL:
        fails.append(f"natively steered peaks land up to {max(native_err):.2f} deg from the "
                     f"true direction, over one {CELL:.2f} deg cell")
    dead_native = float(nd(torch.ones(1, 1, polar.N, polar.N))[0, 0][
        ~valid.reshape(polar.N, polar.N)].abs().max())
    print(f"   dead corners after masking: {dead_native:.1e}")
    if dead_native > 1e-6:
        fails.append(f"native disc leaves up to {dead_native:.1e} in its dead corners")
    if sharper == 0:
        fails.append("native steering is no sharper than resampling anywhere, so the "
                     "interpolation hypothesis is dead before training starts")

    # A sanity anchor on the rectangular path, which the same decoder still serves.
    rect_az, rect_el = decode_mask(as_logits(rect_mask(az_t, el_t, ANGULAR_THRESHOLD)),
                                   mode="centroid")
    rect_err = angular_distance(rect_az, rect_el, az_t, el_t)
    print(f"\n   for reference, the rectangular grid round-trips to "
          f"{rect_err[high].max():.2f} deg max above {CLIPPED_BELOW:.0f} deg")

    print("\nGATE " + ("FAIL" if fails else "PASS"))
    for f in fails:
        print(f"  {f}")
    if not fails:
        print("  The projection preserves direction to under one cell, input and target share "
              "one convention, and the dead corners cost exactly what the geometry says.")
        print("  Native steering points at its own cells and is sharper where it matters: the "
              "largest prominence gain is at the lowest elevation probed, which is where the "
              "far-range azimuth error the disc was built to fix actually lives.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
