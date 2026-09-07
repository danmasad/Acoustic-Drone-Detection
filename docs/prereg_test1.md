# Pre-registration, written before the second pass of runs

Written 25 August 2026, before the queue was launched and therefore before any of the five-cell
results, the DEMAND folds or the all24 K=9 runs existed. Nothing in the prediction section below is
adjusted afterwards; the results go in design notes sections 36 to 41, and the gap between what is
predicted here and what came back is itself a finding. The scorecard at the end was added on
7 September 2026, after everything had run.

## Why this file exists

Test 1 has been scored once. No gradient ever came from it — `utils/unseal_test1.py` only ever
loads `best.pt` and the training-split normalization constants — but we have now *seen* it, and
what it said was specific: classical delay-and-sum beats our U-Net in pooled angular error there,
the failure is concentrated in the near field, and K=9's test-2 advantage does not survive. Any
experiment designed after reading that is conditioned on it, so a second score would carry a bias
whose size nobody can estimate. The leakage channel is us, not the optimizer.

Two facts make this worth handling carefully rather than dogmatically:

1. **There is no replacement split.** The release is 16 WAVs, four flights times four recorders.
   All ten derived segments are assigned and cached; the `ref/` folders hold flight logs for
   flights whose audio was never published. We cannot manufacture a fresh holdout.
2. **Test 1 is underpowered anyway.** 2,210 frames from one flight at one site, and the
   three-seed spread on `hit_10deg` is 3.2 to 7.6 pp. Comparing two three-seed arms, the smallest
   difference that clears twice the standard error of the difference is about 9 pp. Nothing below
   is a 9 pp effect, so test 1 could not arbitrate between these variants even if it were
   pristine.

So the rule for this pass: **test 1 is not touched.** Every number is test 2. If a prediction
below comes true and we later decide test 1 is worth a second look, the prediction was on record
first, which is the only thing that makes a re-used split worth anything.

## Predictions

Confidence is deliberately explicit so that being wrong is legible.

### The five cells, at all24 K=1, on mean azimuth error in the 100-200 m bin

Baseline, already measured: rectangular map with weighted cross-entropy at **57.0 +/- 16.0
degrees**. The resampled disc, already measured, at **82.4 +/- 12.9**.

1. **Native steering beats the resampled disc.** Confidence: moderate-high, around 70%. The gate
   already measured the mechanism rather than assuming it — a synthetic plane wave's peak
   prominence is +1.28 sd higher natively on average and +2.75 sd at 8 degrees elevation, the
   lowest probed, which is where a distant drone sits. Predicted landing zone: 60 to 80 degrees,
   i.e. recovering most of the 25-degree gap the resampling opened.
2. **Native steering does not beat the rectangle.** Confidence: moderate, around 60%. Removing a
   handicap is not the same as winning. The bin holds about 2.5% of drone frames and notes
   section 25 measured that the drone's beamformed peak sits below the ambient's beyond 100 m, so
   a layout change cannot recover a cue the map does not contain.
3. **Tversky moves the far bin more than the layout does.** Confidence: low-moderate, around 40%.
   It is the paper's loss and it weights misses over false positives, which is the right shape
   for a sparse target, but there is no measurement behind this the way there is for the disc.
4. **No cell passes Gate 1** (seed mean at or below 45.0 degrees and every seed below 57.0).
   Confidence: moderate, around 65%. This is the honest base rate: of the hypotheses this project
   has tested since the headline grid, polar, the log-frequency axis and cross-scene detection
   all returned nulls, and the one that worked (deeper history) was the one with a measured
   mechanism behind it beforehand.
5. **Pooled `hit_10deg` separates none of the five cells.** Confidence: high, around 85%. The
   existing two differ by 0.9 pp against seed spreads of 1.8 and 3.6. This is why Gate 1 is
   declared on the far bin and not on the headline metric.

### all24 at K=9

6. **all24 K=9 lands within 1.5 pp of device3 K=9's 73.2%**, in either direction. Confidence:
   moderate, around 60%. Every array-versus-array comparison in this project has come out inside
   seed noise once the recorder count was held fixed, and the six-versus-twenty-four gap was
   -1.2 pp at K=1 and +1.0 at K=5.
7. **all24 K=9 beats all24 K=5's 70.1%.** Confidence: high, around 80%. The freeze-history curve
   on device3 K=9 was still climbing at eight frames, and history helped on every array measured.
8. If 6 holds, the report's hedge becomes a claim: six microphones on one recorder with 0.9 s of
   history match twenty-four synchronized microphones with the same history. If instead all24
   K=9 beats device3 K=9 by more than 1.5 pp, the correct reading is that deeper history rewards
   more microphones too, and the cheap-array claim weakens at K=9. Both are stated now so that
   neither can be presented as the expected result afterwards.

### DEMAND-trained detector

Baseline, already measured: 24,881 false alarms per hour at the 95%-recall threshold, on ambient
it never trained on.

9. **The trained-on groups collapse to a low rate.** Confidence: high, around 85%. Fitting 75
   minutes of ambient it can see is a capacity question and the model has enough.
10. **The held-out group does not** — median across folds stays above 5,000 per hour, so Gate 2
    fails. Confidence: moderate-high, around 70%. Leave-one-scene-out already showed this
    detector generalizes across flights but not across sites (0.488 +/- 0.056 AUROC), and a
    soundscape it has not heard is the same kind of transfer.
11. If 9 and 10 both hold, the finding is sharper than the original: the false-alarm rate is a
    generalization failure and not a training-data shortage, which means more ambient of the same
    kind would not fix it and the per-channel spectrum is the limit.

## What would make test 1 worth a second look

All three of these, not any one:

- a cell passes Gate 1 on test 2 with all three seeds beating the baseline, and
- the same cell holds up at high K on `device3`, against its own 73.2 +/- 1.4 baseline, and
- the improvement is larger than 9 pp on whatever test-1 metric is quoted, which is the smallest
  difference that split can resolve with three seeds.

If those hold, test 1 gets scored once more and the report says plainly that it was a second look
against a prediction registered in this file. If they do not, test 1 keeps the single score it
already has and every new number in the report is test 2.

## Scorecard, added 7 September 2026

Three right, four wrong, two half-right, and the four failures are more useful than the successes.
**Test 1 was not touched**, because the condition for a second look was never met: no cell passed
Gate 1.

| # | Prediction | Conf. | Outcome |
|---|---|---|---|
| 1 | Native steering beats the resampled disc | 70% | **Right**, and by more than predicted: 48.3° against 82.4°, where the landing zone given was 60–80° |
| 2 | Native steering does *not* beat the rectangle | 60% | **Wrong.** 48.3° against the rectangle's 57.0°, so the layout did recover something beyond the handicap it had imposed on itself |
| 3 | Tversky moves the far bin more than the layout does | 40% | **Wrong.** On the rectangle, Tversky made the far bin *worse* (66.4°); the layout moved it and the loss did not |
| 4 | No cell passes Gate 1 | 65% | **Right.** Best cell 45.9° against a 45.0° bar, and no cell beat the baseline on all three seeds |
| 5 | Pooled `hit_10deg` separates none of the five cells | 85% | **Half.** True of the three weighted-CE cells, false for Tversky, which costs 6.6 pp — a real separation we had not anticipated |
| 6 | all24 K=9 lands within 1.5 pp of device3 K=9 | 60% | **Wrong.** 3.1 pp apart, and in the direction that *strengthens* the cheap-array claim rather than weakening it |
| 7 | all24 K=9 beats all24 K=5 | 80% | **Wrong**, and this is the informative one: 70.1% against 70.1%, no gain at all from history on the full array |
| 8 | Conditional reading of 6 | — | The branch that fired was the one favouring the cheap array, so the report's hedge became a sharper claim: depth substitutes for microphones and adds nothing on top of them |
| 9 | Trained-on DEMAND groups collapse to a low rate | 85% | **Half.** A 4.5× cut, but to 5,530/h — still above the 5,000 bar, so even the ambient it trained on is not fitted |
| 10 | Held-out group stays above 5,000/h and Gate 2 fails | 70% | **Right.** 10,384/h |
| 11 | Conditional reading of 9 and 10 | — | Holds, with 9's shortfall making it stronger: the rate is a generalization failure, so more ambient of the same kind would not fix it |

The pattern in the four misses is worth naming. Predictions 2 and 3 were both bets that a
representation change could not help, and prediction 7 was a bet that a mechanism which worked on
one array would work on another. We were most wrong where we assumed a result would transfer, which
is the same lesson the detection experiments taught in a different currency.
