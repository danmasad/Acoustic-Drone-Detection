"""Constants inherited from the dataset, the split definitions, and the pinned
microphone subsets.

"""
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHE = REPO / "cache"
RUNS = REPO / "runs"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

# A run id names one training. The directory around it names the question that training
# answered, so the headline grid, the loss comparison, and the detector stay apart.
_COHERENCE = {"device1", "device4", "device1_3", "symmetric6", "quad4"}
# Superseded runs, kept as evidence for design_notes section 20 and never part of a result.
_ARCHIVED = ("_relu_head", "_deadhead_check")
_DUMP_NAME = re.compile(r"^(?P<run>.+)_(?P<split>test_1|test_2|train)_(?P<order>causal|freeze\d+)$")


def experiment_of(run_id: str) -> str:
    """Which results folder a run belongs in. The first matching test wins.

    _archive: superseded runs kept as evidence (design_notes section 20), never a result.
    detector: the separate drone-present detector.
    loss:     Tversky loss instead of weighted cross-entropy, on either grid.
    disc:     the disc sky grid. _polar resamples the rectangular map onto the
              disc; _polarn steers the beamformer on the disc directly.
    arrays:   one-seed array layouts (device1, device4, device1_3, symmetric6, quad4).
    bins:     8 or 32 frequency bands instead of 16.
    probes:   short test runs and runs named without a width.
    grid:     the headline grid, all24, device3, and best6 at K = 1, 5, and 9.

    A run that uses Tversky on the disc lands in loss, because that test comes first.
    """
    if run_id.endswith(_ARCHIVED):
        return "_archive"
    if run_id.startswith("detector_"):
        return "detector"
    if "tversky" in run_id:
        return "loss"
    if "_polarn" in run_id or "_polar" in run_id:
        return "disc"
    if run_id.split("_K")[0] in _COHERENCE:
        return "arrays"
    if "_b8_" in run_id or "_b32_" in run_id:
        return "bins"
    if "order_probe" in run_id or run_id.endswith("_probe") or "_w32_" not in run_id:
        return "probes"
    return "grid"


def run_dir(run_id: str) -> Path:
    """The folder of one training run: runs/<experiment>/<run_id>/.

    It holds config.json (the settings), metrics.json (the scores and the
    per-epoch history), and best.pt (the weights), for the runs that keep them.
    Every writer and reader of a run goes through this path.
    """
    return RUNS / experiment_of(run_id) / run_id


def pred_file(name: str) -> Path:
    """The per-frame prediction file for one model on one flight.

    Generates the files (cnn_predictions for a trained run, baseline_predictions for delay-and-sum)
    and saves each one at the path returned. Every scoring script opens
    the same path to read it back.

    Accepts the name with or without .npz. Two forms are valid:

    baseline_<array>_<split>      delay-and-sum, no training; filed under
                                  results/preds/baseline/.
    <run_id>_<split>_<order>      a trained network; filed under the same
                                  experiment folder as its run. order is
                                  causal (the normal K-frame input) or freezeJ
                                  (only the newest J frames are real, the
                                  older slots repeat the oldest of those J).

    Each file holds one row per 100 ms frame: the GPS truth, the predicted
    direction, a presence score, and the segment and frame it came from.
    Any other name raises, so nothing is written to a wrong folder.
    """
    stem = name[:-4] if name.endswith(".npz") else name
    if stem.startswith("baseline_"):
        return RESULTS / "preds" / "baseline" / f"{stem}.npz"
    match = _DUMP_NAME.match(stem)
    if match is None:
        raise ValueError(f"dump name {name!r} is not baseline_* or {{run}}_{{split}}_{{order}}")
    return RESULTS / "preds" / experiment_of(match.group("run")) / f"{stem}.npz"


def table_file(name: str) -> Path:
    """A summary JSON, filed by the question it answers.

    Each summary table holds the aggregated results from a prediction file:
    hit rates by range, seed means and spreads, confidence intervals, false alarms. 
    The analysis scripts compute them from the prediction files
    or from each run's metrics.json and save them at the path returned here.
    """
    if name.startswith("history_") or name == "paired_k.json":
        group = "history"
    elif name.startswith("polar_"):
        group = "disc"
    elif name.startswith("demand_") or name.startswith("detector_loso_"):
        group = "detector"
    else:
        group = "scores"
    return RESULTS / "tables" / group / name


def search_file() -> Path:
    """The geometric microphone-subset search, results/search/subset_search.npz.

    Every non-coplanar 4- and 6-microphone subset (2,142 and 106,183) is scored
    on 250 simulated directions with perfect clocks and plane waves: the level
    of the strongest false peak more than 15 degrees off the source, in dB below
    the true peak.
    """
    return RESULTS / "search" / "subset_search.npz"

# --- inherited from the dataset and create_dataset.py ---------------------------------
SR = 48000
CHUNK = 4800                     # 100 ms frames, non-overlapping, as in create_dataset.py
FRAME_SECONDS = CHUNK / SR
C = 343.0                        # speed of sound, m/s
N_CHANNELS = 24                  # four 6-channel recorders
CH_PER_DEVICE = 6
BYTES_PER_SAMPLE = 4             # int32 PCM

# GPS logs are 10 Hz, i.e. exactly one row per audio frame. Never resample; index them.
CSV_HZ = 10

# UTM zone 56 south covers both recording sites (Western Sydney). The dataset's own
# scripts project into it, so we inherit the choice rather than picking a local frame.
UTM_EPSG = "epsg:32756"
FEET_TO_M = 0.3048
MPH_TO_MPS = 0.44704

# --- direction grid (4 degree spacing) ----------------------------------
AZ = np.arange(-180, 181, 4)     # 91 values, circular axis
EL = np.arange(0, 91, 4)         # 23 values
GRID_SHAPE = (len(EL), len(AZ))  # (23, 91): elevation is rows, azimuth is columns

# --- frequency handling --------------------------------------------------------------
# Cache wide at the frame's native 10 Hz resolution, feed the model narrow. Caching more
# than the model needs is what keeps the frequency-bin ablation and any band change a
# runtime flag instead of another pass over 24 GB of audio.
CACHE_BAND = (200.0, 4000.0)     # 381 native bins
MODEL_BAND = (200.0, 2200.0)     # the paper's band
N_SUBBANDS = 16                  # our reconstruction of their F=16


def run_id(subset: str, K: int, *, base: int = 32, subbands: int = N_SUBBANDS,
           loss: str = "wce", polar: bool = False, disc: str = "resample",
           seed: int | str | None = 0, tag: str = "") -> str:
    """The on-disk name of one training: <subset>_K<K>_w<base>_b<subbands>_<loss>[_polar|_polarn]_s<seed>[_<tag>].

    `seed` may be a template such as "{s}" for scripts that fill it per seed, or None to
    get the seedless prefix. The resampled disc keeps the older `_polar` suffix so runs
    already on disk are not retrained; the natively steered disc is `_polarn`. `src.train`
    and `utils/train/run_grid.py` write this same format and must stay in step with it.
    """
    layout = "" if not polar else ("_polarn" if disc == "native" else "_polar")
    seed_part = "" if seed is None else f"_s{seed}"
    tag_part = f"_{tag}" if tag else ""
    return f"{subset}_K{K}_w{base}_b{subbands}_{loss}{layout}{seed_part}{tag_part}"


assert run_id("all24", 1, seed=2) == "all24_K1_w32_b16_wce_s2"
assert run_id("all24", 1, polar=True, disc="native") == "all24_K1_w32_b16_wce_polarn_s0"
assert run_id("device3", 9, seed="{s}") == "device3_K9_w32_b16_wce_s{s}"

# --- evaluation ----------------------------------------------------------------------
ANGULAR_THRESHOLD = 10           # the dataset's own label tolerance, in degrees
HIT_THRESHOLDS = (5, 10, 20)     # strict / inherited / operational

# --- microphone subsets -------------------------------------------------------------
# Pinned explicitly, never recomputed at run time: the array has 3-fold symmetry so
# several subsets tie exactly, and a re-run could silently return a mirror image.
# Sidelobe figures are the 95th percentile worst false peak over 250 source directions
# from the dense search in results/search/subset_search.npz.
_DEV = {k: list(range(CH_PER_DEVICE * k, CH_PER_DEVICE * (k + 1))) for k in range(4)}

MIC_SUBSETS = {
    "all24":       list(range(24)),                    # -3.6 dB, ceiling
    "best6":       [12, 13, 14, 19, 21, 23],           # -3.6 dB, searched optimum, ties all24
    "device3":     _DEV[2],                            # -2.8 dB, one recorder, no sync: the claim
    "symmetric12": [0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17],   # -2.8 dB
    "device1_3":   _DEV[0] + _DEV[2],                  # -2.4 dB, 12 mics yet worse than device3
    "symmetric6":  [0, 1, 2, 15, 16, 17],              # -1.8 dB
    "device2":     _DEV[1],                            # -1.7 dB
    "device1":     _DEV[0],                            # -1.5 dB, worst spread at n=6
    "greedy6":     [2, 9, 11, 15, 16, 17],             # -1.2 dB, maximally aliased stress case
    "device4":     _DEV[3],                            # -0.2 dB, coplanar, worst of all 106,183
    "quad4":       [12, 13, 14, 22],                   # -1.8 dB, representative 4-mic floor
}

SIDELOBE_DB = {
    "all24": -3.6, "best6": -3.6, "device3": -2.8, "symmetric12": -2.8,
    "device1_3": -2.4, "symmetric6": -1.8, "device2": -1.7, "device1": -1.5,
    "greedy6": -1.2, "device4": -0.2, "quad4": -1.8,
}

# --- sessions and segments ----------------------------------------------------------
SESSION_DIRS = [
    DATA / "training/Nov_25_2024/flight_11",
    DATA / "training/Mar_18_2025/flight_1",
    DATA / "test_2/Mar_18_2025/flight_2",
    DATA / "test_1/Oct_11_2024/flight_5",
]

# The trap this guards against: the only no-drone audio in the dataset is the tail of
# test_2's flight_2, and half of it is training data. Split membership therefore comes
# from the segment's own role, not from the directory it sits in.
_SPLIT_OF_ROLE = {
    "training": "train",
    "training (negative class)": "train",
    "test_1": "test_1",
    "test_1 (negative class)": "test_1",
    "test_2": "test_2",
    "test_2 (negative class)": "test_2",
}

# Every session starts recording before the flight, so each WAV opens with minutes of
# genuine no-drone audio that the dataset's own splits never use. Mining it more than
# doubles the negative set and, more importantly, spreads it over four sessions instead of
# one 5-minute tail. Verified ambient rather than drone by beamformed peak contrast
# (utils/validate/probe_pretakeoff.py); a segment inherits the split of its own session, which keeps
# test audio out of training by construction.
PRE_TAKEOFF_START_S = 0.2      # skip the very first samples: recorder corrections are negative
PRE_TAKEOFF_MARGIN_S = 20.0    # drop the run-up to takeoff: motor spin-up, handling, chatter


class Segment:
    """One continuous piece of one flight, with one label regime.

    Built from the downloaded alignment JSON. A flight either lists its pieces
    or stores the same fields at the top level, so each field is read from the
    piece and falls back to the flight record.

    session_dir, session, and flight_id locate the audio. name distinguishes
    the drone piece from a silent one. split is train, test_1, or test_2.
    skip_seconds and skip_seconds_end bound the piece inside the WAV.
    flight_csv is the GPS log, and is absent when the piece has no drone.
    start_index_csv and initial_azimuth_deg line that log up with the audio.
    align keeps the whole JSON for the microphone positions.

    id is the cache folder name. cache_dir is cache/ and that name.
    has_drone is true when a GPS log is present.
    """

    def __init__(self, session_dir: Path, spec: dict, align: dict):
        # A session either declares segments explicitly or is one implicit segment whose
        # fields sit at the top level, so every lookup falls back from spec to align.
        def field(key, default=None):
            return spec.get(key, align.get(key, default))

        self.session_dir = session_dir
        self.session = align["session"]
        self.flight_id = align["flight_id"]
        self.name = spec.get("name", "drone")
        self.split = _SPLIT_OF_ROLE[field("role")]
        self.skip_seconds = float(field("skip_seconds"))
        self.skip_seconds_end = field("skip_seconds_end")
        self.flight_csv = field("flight_csv")
        self.start_index_csv = field("start_index_csv")
        self.initial_azimuth_deg = field("initial_azimuth_deg")
        self.align = align

    @property
    def id(self) -> str:
        return f"{self.session}_f{self.flight_id}_{self.name}"

    @property
    def has_drone(self) -> bool:
        return self.flight_csv is not None

    @property
    def cache_dir(self) -> Path:
        return CACHE / self.id

    def __repr__(self) -> str:
        return f"<Segment {self.id} split={self.split} drone={self.has_drone}>"


def load_align(session_dir: Path) -> dict:
    """One flight's alignment_params.json as a dict: microphone positions, clock corrections,
    audio start offset, and the split the flight belongs to."""
    return json.loads((session_dir / "alignment_params.json").read_text())


def _pre_takeoff_spec(align: dict, specs: list[dict]) -> dict | None:
    """The no-drone window before the flight starts, or None if there is too little of it.

    The returned piece has no GPS file. Every 100 ms frame is therefore labelled
    no drone: has_drone is 0, azimuth, elevation, range, and speed are missing,
    and the direction grid is full of zeros. The presence target is
    absent. The beamformed audio is still fed to the network.
    """
    flight_skip = min(float(s.get("skip_seconds", align.get("skip_seconds", 0.0)))
                      for s in specs if s.get("flight_csv", align.get("flight_csv")))
    end = flight_skip - PRE_TAKEOFF_MARGIN_S
    if end - PRE_TAKEOFF_START_S < 5.0:
        return None
    return {"name": "pre_takeoff", "role": f"{align['role']} (negative class)",
            "skip_seconds": PRE_TAKEOFF_START_S, "skip_seconds_end": end,
            "flight_csv": None}


def segments(split: str | None = None) -> list[Segment]:
    """Every segment we train and test on, in a fixed order.

    Sessions declare either a single implicit segment at the top level or an explicit
    `segments` list; both are flattened here so callers never special-case the layout.
    """
    out = []
    for d in SESSION_DIRS:
        align = load_align(d)
        specs = align.get("segments") or [{}]
        out += [Segment(d, s, align) for s in specs]
        pre = _pre_takeoff_spec(align, specs)
        if pre is not None:
            out.append(Segment(d, pre, align))
    return [s for s in out if split is None or s.split == split]
