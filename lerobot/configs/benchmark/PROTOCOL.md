# Evaluation protocol

The evaluation contains four tasks, 320 failure recordings, and 80 successful
test recordings. Each task uses 19 separate successful policy rollouts for
calibration and 20 for testing false alarms. Failure modes and natural/induced
origins describe different attributes of the same recordings.

Scores are unsmoothed and an alarm requires a strict threshold crossing.
FDR counts failures with a crossing at or after the view-specific mean onset
from three annotations. FAR counts successful test recordings with any crossing.
Counts are pooled across tasks; episode AUROC uses whole-episode maxima and
averages task AUROCs equally. Calibration uses no failure labels.

The constant threshold is the maximum of 19 calibration-episode maxima.
Time-varying bands use frozen expert-derived templates. The **5% FAR budget**
comparison chooses retrospective thresholds from successful test recordings;
it is distinct from calibration and is not a deployment guarantee.

Continuous-method detection rates use offline scores. Qwen uses full-test
streaming decisions and includes late returned detections in its FDR.
Latency uses on-time decisions, measured from onset to decision completion.
Paired comparisons use each pair's shared detected recordings and each camera's
own onset. These conditional latencies are not full-test online detection rates.

logpZO and RND **Policy** use ACT-native features, as in the paper. **VAE** rows
are supplementary variants using the frozen visual latents. Both logpZO
variants retain proprioception. NoACT removes action conditioning from WM-NC.

Calibration sensitivity uses all 500 saved 19/20 success splits with fixed
failures. The four-seed summary pools task counts within each seed before
computing the mean and sample SD. Split statistics describe overlapping
resamples, not independent experiments or confidence intervals.

Fusion records explicitly distinguish endpoint and episode aggregation.
Endpoint AND requires simultaneous alarms; episode AND allows different
eligible post-onset endpoints. The paper's 45.6% AND FDR corresponds to episode
aggregation. OR has the same result under either aggregation on common support.

The downloadable CSV retains exact counts and rates. Rates are fractions unless
a column says otherwise; blank cells are unavailable, not zero. Displayed values
are rounded, so counts should be used for recomputation.

## Reproduction

The machine-readable specification is [evaluation_protocol_v3.yaml](evaluation_protocol_v3.yaml).
[Failure categories](failure_categories.yaml) and [dataset validity](dataset_corrections.yaml)
are versioned separately. The valid support for Needle Pickup recording 5 is
raw frames 0–121 in both views; all detectors use the same mask.

Successful expert demonstrations number 250 per simulation task and 100 per
real task. ACT uses all demonstrations. Visual VAEs and the main world model
share episode splits of 225/25 and 90/10 for training/validation, respectively.
The visual backbone is frozen DINOv2 ViT-B/14 with registers. The main world
model uses K = L = 4 and one RK4 integration step per score.

[Fusion definitions](FUSION_PROTOCOL.md) specify common endpoint support.
[Public result tables](../../../docs/results/README.md) contain the current
measurements. Local archives retain the full input hashes and run provenance.
