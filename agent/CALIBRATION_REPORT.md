# MAZU Extension — Reliability Diagrams (Calibration Curves)

## Motivation

Every metric reported so far answers a different question than "is the probability
itself trustworthy":
- ROC-AUC/PR-AUC: does the model **rank** true events above non-events (threshold-independent)?
- POD/FAR/CSI/HSS: at one **fixed operational threshold**, how many alerts are hits vs. false alarms?

Neither answers: **when the model says "70% risk," does the event actually happen ~70% of
the time?** That is a calibration question, and it is what a reliability diagram measures.

## What it does

`model/09_calibration.py` bins every held-out test-set prediction (Jul–Dec 2025, the
same split used everywhere else) into 10 equal-width probability buckets [0,0.1),
[0.1,0.2), ..., [0.9,1.0], for each of the 3 hazards. For each bucket it computes the
mean predicted probability and the actual observed event frequency, and plots both
against a diagonal "perfect calibration" reference line. It also computes two summary
numbers: **ECE** (Expected Calibration Error — count-weighted average gap between
predicted and observed) and **Brier score** (mean squared error of the probability
itself). Like the other evaluation extensions, this uses the **already-saved production
models**, never retrained, so the numbers cannot drift from what the agent actually serves.

## The real finding — and why the headline ECE numbers alone would be misleading

| Hazard | ECE | Brier | Top bin [0.9,1.0]: predicted vs. observed |
|---|---|---|---|
| Heatwave | 0.039 | 0.030 | 0.969 vs. 0.770 |
| Flash flood | 0.006 | 0.006 | 0.957 vs. 0.543 |
| Dust storm | 0.085 | 0.060 | 0.949 vs. 0.204 |

Read in isolation, flash flood's ECE (0.006) looks like the *best-calibrated* of the
three hazards — but this is misleading, and disclosing only that number would
oversell the model. **~97% of flash-flood test samples fall in the [0,0.1) bucket**,
where the model is genuinely well-calibrated (predicted ≈0.5%, observed ≈0.4%) — this
single dominant bucket pulls the count-weighted ECE average down to near zero and
masks what happens elsewhere.

Looking at the reliability diagram itself (not just the summary number) shows every
single bucket, for **all 3 hazards**, falls **below** the diagonal — i.e. the model is
**systematically overconfident** whenever it does predict elevated risk, worst at the
high end: for flash flood, a "95.7% probability" bucket only sees the event 54.3% of
the time; for dust storm, a "94.9% probability" bucket only sees it 20.4% of the time.

**This is directly relevant to CAP alert severity** (§ `agent/CAP_REPORT.md`): an
"Extreme" severity CAP alert (probability ≥0.85) is issued off a probability number
that this analysis shows is itself somewhat overconfident at that range. This does not
invalidate the alert-issuance logic (POD/FAR/CSI/HSS are still computed and reported
honestly at the operational threshold), but it is a genuine, disclosed limitation of
the raw probability's face-value interpretation, found by looking past the single
aggregate ECE number rather than stopping there.

## Testing

`model/09b_test_calibration.py` — 24 checks, independent of the main script's own
output: 3 synthetic, hand-verifiable cases (a perfectly-calibrated-by-construction
model; a hand-constructed overconfident model with an exactly-known 0.9→0.3 gap,
checked against the hand-computed formula; a sparse case confirming empty bins are
reported as `count: 0` with `None` values rather than silently dropped or fabricated
as zero) plus structural checks against the real `calibration_report.json` (bin counts
sum to the full test-set size, ECE/Brier are valid [0,1] values) and an explicit,
failing-if-wrong assertion that all 3 hazards' top bin shows genuine overconfidence —
locking in the finding above rather than only eyeballing the chart.

`FULL_SYSTEM_AUDIT.py` Section N (5 checks, part of the full audit total) independently
rebuilds flash flood's held-out test set from the raw dataset, loads the saved model,
and recomputes the bins/ECE/Brier completely from scratch — bypassing both `tools.py`
and `09_calibration.py`'s own stored report — and confirms the overconfidence finding
a second time, independent of the first computation.

## Follow-up: is the overconfidence fixable? Tested, yes — not deployed

`model/10_calibration_fix.py` tests whether standard isotonic-regression
recalibration actually fixes the problem, using a leak-free **3-way**
chronological split so the test set is never touched by fitting anything:

- **Jan 1 – May 31**: trains an *isolated* classifier (same hyperparameters
  as production, but less data — June is deliberately withheld so it is
  genuinely unseen).
- **June 1–30**: the isolated classifier's own predictions here are used to
  fit an isotonic-regression calibrator. Using June for calibration only
  works because the classifier never trained on it — fitting a calibrator
  on data the classifier already memorized would understate its real error.
- **Jul 1 – Dec 31**: the same, unchanged test set used everywhere else in
  this project — touched by neither the classifier's training nor the
  calibrator's fitting, used only for the final before/after comparison.

| Hazard | Brier before → after | ECE before → after | ROC-AUC before → after |
|---|---|---|---|
| Heatwave | 0.0273 → 0.0265 | 0.014 → **0.023 (worse)** | 0.9548 → 0.9551 |
| Flash flood | 0.011 → 0.0049 | 0.022 → 0.003 | 0.9039 → 0.8974 |
| Dust storm | 0.0643 → 0.0193 | 0.099 → 0.014 | 0.8018 → 0.8010 |

Brier score (the metric a probability calibrator is actually optimizing)
**improved for all 3 hazards**, most dramatically for dust storm and flash
flood — confirmed visually too (the reliability-diagram points move from
well below the diagonal to hugging it closely).

**Two further real, disclosed findings from this experiment, not smoothed over:**

1. **Heatwave's ECE got slightly *worse*** (0.014 → 0.023) even though its
   Brier score improved. This is not a contradiction — it is the same
   lesson as flash flood's misleadingly-low aggregate ECE above: a single
   scalar metric can disagree with another equally valid one, which is
   exactly why this project reports several metrics together rather than
   picking whichever looks best.
2. **ROC-AUC is only *approximately* preserved, not exactly unchanged.**
   The initial assumption ("isotonic regression is monotonic, so ranking
   is untouched") turned out to be imprecise: isotonic regression is
   monotonic *non-decreasing*, not strictly increasing — it collapses runs
   of calibration-set probabilities into flat steps, so distinct raw
   test-set probabilities can land on an identical calibrated value,
   introducing ties that were not present in the original ranking. Flash
   flood (fewest positive samples, so each new tie matters more) shows the
   largest real shift: ROC-AUC 0.9039 → 0.8974. Small, but real and
   measured — not assumed to be exactly zero.

### Why this is NOT deployed to the production models

Swapping calibrated probabilities into `forecast_tool` would change the
numeric output of essentially every existing verified claim: the specific
probability values asserted in ~150 existing unit/audit checks (e.g.
`0.9416` for a known Jizan event), and — more importantly — **CAP alert
severity** (`agent/CAP_REPORT.md`) is deliberately mapped from
`DetectionEngine`'s own severity thresholds (0.50/0.55/0.70/0.85) tuned to
the *raw*, uncalibrated probability scale; recalibrating the underlying
probabilities would require re-deriving those thresholds too, and
re-verifying every downstream test that depends on them. Given the
isolated-classifier caveat above (trained on 1 less month of data than
production, so its raw numbers are not directly comparable to the
production models' own reported metrics), this is kept as a **documented,
tested research finding** — the same treatment given to the GNN variant and
the neighbor-mean feature earlier in this project: real, honestly reported,
not deployed.

## Testing

`model/09b_test_calibration.py` — 24 checks, independent of the main script's own
output: 3 synthetic, hand-verifiable cases (a perfectly-calibrated-by-construction
model; a hand-constructed overconfident model with an exactly-known 0.9→0.3 gap,
checked against the hand-computed formula; a sparse case confirming empty bins are
reported as `count: 0` with `None` values rather than silently dropped or fabricated
as zero) plus structural checks against the real `calibration_report.json` (bin counts
sum to the full test-set size, ECE/Brier are valid [0,1] values) and an explicit,
failing-if-wrong assertion that all 3 hazards' top bin shows genuine overconfidence —
locking in the finding above rather than only eyeballing the chart.

`model/10b_test_calibration_fix.py` — 26 further checks: confirms the 3-way split is
genuinely leak-free (exact date-range assertions on each of the 3 periods), that
before/after bin counts conserve the full test-set sample count, that Brier
genuinely improves for all 3 hazards, that heatwave's ECE regression is explicitly
asserted (not smoothed over), that the ROC-AUC shift is real-but-small (tolerance-
based, with an explicit non-zero assertion for flash flood so a "silently rounds to
0" bug can't hide), and a full bypass that independently re-trains the isolated
classifier and re-fits the isotonic calibrator from raw data, reproducing the
stored Brier score exactly (deterministic given the fixed `random_state=42`).

`FULL_SYSTEM_AUDIT.py` Section N (part of the full audit total) independently
rebuilds flash flood's held-out test set from the raw dataset, loads the saved model,
and recomputes the bins/ECE/Brier completely from scratch — bypassing both `tools.py`
and `09_calibration.py`'s own stored report — and confirms the overconfidence finding
a second time, independent of the first computation.

## Performance impact

None on the model/KG/agent/CAP layers — the fix experiment trains a completely
separate, isolated classifier instance purely for this analysis. `09_calibration.py`
and `10_calibration_fix.py` only read/train models outside the live agent path;
nothing `forecast_tool` or `cap_alert_tool` serve was touched. All existing unit
tests and audit checks pass unchanged.
