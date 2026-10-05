#!/usr/bin/env bash
# The whole remaining compute queue, in one sequential chain.
#
# Sequential because MPS is one compute unit: two runs at once each take twice as long. Ordered
# cheapest-and-most-decisive first, so that stopping early still leaves a coherent set of
# results rather than a half-finished experiment. Every step skips itself if its output already
# exists, so the chain is resumable after a stop or a reboot.
#
#   .venv/bin/python utils/setup/detach.py logs/remaining.log -- bash utils/train/run_remaining.sh
set -uo pipefail                       # not -e: one failed stage must not sink the queue
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

step () { echo; echo "=== $* ==="; date "+%H:%M:%S"; }

# --- detector stages, minutes each ---------------------------------------------------------
# A6 first: it is the protocol every later detector number is judged under, so it has to exist
# before any of them mean anything.
step "A6  leave-one-scene-out, linear axis, no confusers"
$PY -u utils/analysis/detector_loso.py

step "C   leave-one-scene-out, log-frequency axis (64 bins)"
$PY -u utils/analysis/detector_loso.py --log-bins 64

step "B   leave-one-scene-out, ESC-50 confusers mixed in"
$PY -u utils/analysis/detector_loso.py --confusers

step "B+C leave-one-scene-out, both"
$PY -u utils/analysis/detector_loso.py --confusers --log-bins 64

# Trained on all three of our scenes, which is the model to quote a deployment number for,
# since DEMAND is unseen either way and holding a scene out only weakens it.
step "full-train detectors, for the DEMAND false-alarm measurement"
det () {                               # det <run_id> <flags...>
    if [ -f runs/*/"$1"/metrics.json ]; then echo "skip $1, already trained"; return; fi
    shift; $PY -u -m src.detector --tag full "$@"
}
det detector_K1_w32_s0_full
det detector_K1_w32_log64_s0_full --log-bins 64
det detector_K1_w32_conf_s0_full --confusers
det detector_K1_w32_conf_log64_s0_full --confusers --log-bins 64

# The false-alarm rate the report quotes, on 90 minutes of ambient from 18 environments none of
# these models has heard. Threshold comes from our own data; nothing here changes it.
step "DEMAND false alarms per hour, per scene"
$PY -u utils/analysis/demand_false_alarms.py detector_K1_w32_s0_full
$PY -u utils/analysis/demand_false_alarms.py detector_K1_w32_conf_s0_full

# --- map-model batteries, hours ------------------------------------------------------------
step "E   polar map, all24 K=1, three seeds"
$PY -u utils/train/run_grid.py polar

step "F   sub-band sweep, device3 K=1, 8 and 32"
$PY -u utils/train/run_grid.py bins

# Gated on utils/analysis/check_history_depth.py, which passed: accuracy was still climbing at five
# frames (+0.53 pp on the last step, 7 standard errors), so nine frames is worth measuring.
step "G   K=9 on device3, three seeds"
$PY -u utils/train/run_grid.py k9

# K=9 won (+2.1 pp over K=5), so it earns two things: a place in the sealed-split scoring, and the
# same depth intervention applied to its own weights, which is what locates the saturation the K=5
# curve could only bound from below. Dumps are skipped if the npz already exists.
step "G'  K=9 per-frame dumps at every history depth, then the depth curve"
$PY -u -m src.predict --split test_2 \
    --runs device3_K9_w32_b16_wce_s0 device3_K9_w32_b16_wce_s1 device3_K9_w32_b16_wce_s2 \
    --orders causal freeze1 freeze2 freeze3 freeze4 freeze5 freeze6 freeze7 freeze8
$PY -u utils/analysis/check_history_depth.py --K 9 --subsets device3

# --- H: test 1, once ------------------------------------------------------------------------
# Reads only checkpoints that already exist, gates polar and K=9 on whether they beat their
# counterparts on test 2, and refuses to overwrite results/tables/scores/test_1.json without
# --force. Every choice in the repository was made before this line ran.
step "H   unseal test 1 and score everything"
$PY -u utils/analysis/unseal_test1.py

step "queue finished"
date "+%H:%M:%S"
