"""Constants inherited from the dataset, the split definitions, and the pinned
microphone subsets.

Nothing here is a free choice except MIC_SUBSETS and the band/sub-band settings.
Everything else is fixed by the dataset or by the paper we are comparing against, so it
lives in one place and is never re-derived locally.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHE = REPO / "cache"
RUNS = REPO / "runs"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

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

# --- direction grid (the paper's, 4 degree spacing) ----------------------------------
AZ = np.arange(-180, 181, 4)     # 91 values, circular axis
EL = np.arange(0, 91, 4)         # 23 values
GRID_SHAPE = (len(EL), len(AZ))  # (23, 91): elevation is rows, azimuth is columns

# --- frequency handling --------------------------------------------------------------
# Cache wide at the frame's native 10 Hz resolution, feed the model narrow. Caching more
# than the model needs is what keeps the frequency-bin ablation and any band change a
# runtime flag instead of another pass over 24 GB of audio.
CACHE_BAND = (200.0, 4000.0)     # 381 native bins
MODEL_BAND = (200.0, 2200.0)     # the paper's band
N_SUBBANDS = 16                  # our reconstruction of their F=16 (design_notes section 16)

# --- evaluation ----------------------------------------------------------------------
ANGULAR_THRESHOLD = 10           # the dataset's own label tolerance, in degrees
HIT_THRESHOLDS = (5, 10, 20)     # strict / inherited / operational

# --- microphone subsets -------------------------------------------------------------
# Pinned explicitly, never recomputed at run time: the array has 3-fold symmetry so
# several subsets tie exactly, and a re-run could silently return a mirror image.
# Sidelobe figures are the 95th percentile worst false peak over 250 source directions
# from the dense search in results/subset_search.npz (design_notes section 14).
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
# (utils/probe_pretakeoff.py); a segment inherits the split of its own session, which keeps
# test audio out of training by construction.
PRE_TAKEOFF_START_S = 0.2      # skip the very first samples: recorder corrections are negative
PRE_TAKEOFF_MARGIN_S = 20.0    # drop the run-up to takeoff: motor spin-up, handling, chatter


class Segment:
    """One contiguous stretch of audio with a single label regime."""

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
    return json.loads((session_dir / "alignment_params.json").read_text())


def _pre_takeoff_spec(align: dict, specs: list[dict]) -> dict | None:
    """The no-drone window before the flight starts, or None if there is too little of it."""
    flight_skip = min(float(s.get("skip_seconds", align.get("skip_seconds", 0.0)))
                      for s in specs if s.get("flight_csv", align.get("flight_csv")))
    end = flight_skip - PRE_TAKEOFF_MARGIN_S
    if end - PRE_TAKEOFF_START_S < 5.0:
        return None
    return {"name": "pre_takeoff", "role": f"{align['role']} (negative class)",
            "skip_seconds": PRE_TAKEOFF_START_S, "skip_seconds_end": end,
            "flight_csv": None}


def segments(split: str | None = None, mined_negatives: bool = True) -> list[Segment]:
    """Every segment we train and test on, in a fixed order.

    Sessions declare either a single implicit segment at the top level or an explicit
    `segments` list; both are flattened here so callers never special-case the layout.
    `mined_negatives=False` reproduces the paper's splits exactly, which is what the
    with-and-without detection comparison needs.
    """
    out = []
    for d in SESSION_DIRS:
        align = load_align(d)
        specs = align.get("segments") or [{}]
        out += [Segment(d, s, align) for s in specs]
        pre = _pre_takeoff_spec(align, specs) if mined_negatives else None
        if pre is not None:
            out.append(Segment(d, pre, align))
    return [s for s in out if split is None or s.split == split]


# Scene = one recording session and flight at site 1, by segment id. The unit of
# leave-one-scene-out for the detector (notes 28) and of leave-one-flight-out for the map model.
# Scene C holds all of test 2 plus the half of its ambient tail that the dataset calls training.
SCENES = {
    "A_Nov25": ["Nov_25_2024_f11_drone", "Nov_25_2024_f11_pre_takeoff"],
    "B_Mar18_f1": ["Mar_18_2025_f1_drone", "Mar_18_2025_f1_pre_takeoff"],
    "C_Mar18_f2": ["Mar_18_2025_f2_drone", "Mar_18_2025_f2_no_drone_1",
                   "Mar_18_2025_f2_no_drone_2", "Mar_18_2025_f2_pre_takeoff"],
}


def scene_segments(names) -> list[Segment]:
    """The segments of the named scenes, in SCENES order; unknown ids fail loudly."""
    lookup = {seg.id: seg for split in ("train", "test_2") for seg in segments(split)}
    missing = [sid for n in names for sid in SCENES[n] if sid not in lookup]
    if missing:
        raise KeyError(f"unknown segment ids {missing}")
    return [lookup[sid] for n in names for sid in SCENES[n]]
