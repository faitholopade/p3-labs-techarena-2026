# How the solution works

The submitted predictions came from an ensemble built over successive versions.
The older inference modules remain because later submissions depend on them.

## User preferences and exposure

The training data records both the ten displayed items and the clicked item,
if there was one. We used those signals separately: clicks describe preferences,
while displays also tell us about the candidate-selection process.

The core implementations are [variational.py](../src/variational.py),
[residual_model.py](../src/residual_model.py) and
[train_factor_core.py](../src/train_factor_core.py). They fit user and item
representations, uncertainty parameters and exposure-related effects. Prior
clicks from the warm-up period supply additional preference information.
[posterior_inference.py](../src/posterior_inference.py) integrates sampled latent
scores into candidate probability and uncertainty features.

## Features and calibration

[features.py](../src/features.py) builds preference and history features.
[activity.py](../src/activity.py) and
[v25_exposure_memory.py](../src/v25_exposure_memory.py) add activity and exposure
memory. The historical code also includes tree models and several neural heads.

[calibration_model.py](../src/calibration_model.py) fits a head on top of frozen
features and offsets. Its event loss uses ten item logits plus a no-click logit.
Some variants add a pairwise ranking loss. This matters because an event can
have a plausible item ranking but an incorrect overall click probability.

## V63 and V65

[scripts/research_v63.py](../scripts/research_v63.py) fine-tunes a head on first
queries of the day. The aim was to better match the evaluation setting.

[event_mass_correction.py](../src/event_mass_correction.py) is the V65 event
correction. It uses event-level history/activity features and a frozen event
logit to learn a bounded change in click propensity. Applying it rescales the
candidate probabilities together, preserving their relative proportions.
[scripts/research_v65.py](../scripts/research_v65.py) contains its fitting and
validation stages; [infer_after_v63.py](../src/infer_after_v63.py) applies the
release recipe after reconstructing its parents.

## V142

[V141 research](../experiments/v141/research.py) fits a weekend preference
correction using pre-cutoff clicks and warm-up history. It includes nuisance
terms for source and time bucket, plus a shared or hour-specific weekend term.
The selected V142 recipe uses the shared term with strength 0.5.

[V142 replay](../experiments/v142/replay.py) contains the exact saved inference
calculation. It standardizes catalog features, clips them to [-6, 6], centers
them within the slate, and projects onto the learned weekend weights. The
adjustment is bounded by `0.2 * tanh(linear / 0.2)` and added to the log baseline
probabilities at half strength. A softmax redistributes the original event's
click mass among its items. No-click probability is therefore unchanged.

[confirm.py](../experiments/v142/confirm.py) checks the chosen recipe on the reused
validation cohorts. [build.py](../experiments/v142/build.py) creates the
production replay inputs, checks alignment, probability bounds and candidate
permutation, and writes the submission CSV/ZIP. The separate research-archive bundler is not included.

The baseline is saved V136 predictions. That dependency is important: the V142
replay is not a complete training pipeline, even though its final correction
code is short. [Running the code](running.md) explains what is available.
