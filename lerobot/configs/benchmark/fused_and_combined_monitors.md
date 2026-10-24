# Fused-view models and combined monitors

**Fused-view models** concatenate fixed and wrist latent channels in a single
K4/L4 world-model branch. They use one joint endpoint score and threshold.
WM-NC and WM-PE sum coordinate squared norms/errors across both views; WM-PU
averages coordinate standard deviations across the concatenated latent.
Per-view contributions from this model are not independent detectors.

**Combined monitors** combine separately calibrated detectors: WM-NC across
fixed and wrist views, or WM-NC with Policy logpZO within a view. The alarm
margin is `(score - threshold) / scale`, with a strict `> 0` crossing.
Endpoint OR uses the maximum margin and endpoint AND uses the minimum at the
same raw frame. Episode AND instead requires each detector to cross within
the eligible post-onset interval, potentially at different frames. See the
[fusion protocol](FUSION_PROTOCOL.md) for these distinct definitions.

Component thresholds and scales are fitted only on calibration successes.
OR can increase the false-alarm rate; it does not inherit an individual
detector's 5% guarantee. The retrospective 5% FAR budget analysis provides
a separate comparison at a common budget.

Online method fusion evaluates both resident encoders/scorers sequentially
at one selected endpoint and charges both costs. View fusion shares a world-
model call. Raw images are preloaded into CPU memory; visual features are
computed during replay. Conditional replay selects offline-detected failures
and therefore reports conditional latency and retention, not full-test
online FAR/FDR.

Model configurations and training commands are described in the
[fused-view architecture guide](../view_fusion_ablation.md). Joint scores are
formed at each endpoint before calibration; separate episode maxima or
thresholds must not be added to construct a joint detector.
