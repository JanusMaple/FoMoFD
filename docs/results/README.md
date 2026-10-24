# Results

[Evaluation protocol](protocol.md)

Current results use the three-annotation mean onset and the same frozen
evaluation population as the revised paper. Rates pool episode counts.
The website includes supplementary VAE-feature baselines, labeled separately
from the ACT-native Policy baselines used in the paper.

| Data | Contents |
| --- | --- |
| [offline_by_failure.csv](details/offline_by_failure.csv) | Detection counts, calibration and 5% FAR budget results |
| [online_by_failure.csv](details/online_by_failure.csv) | Streaming outcomes and conditional replay summaries |
| [pairwise_latency_summary.csv](details/pairwise_latency_summary.csv) | On-time paired latency comparisons |
| [online_episode_outcomes.csv](details/online_episode_outcomes.csv) | Individual measured online outcomes |
| [pairwise_latency_episodes.csv](details/pairwise_latency_episodes.csv) | Shared recordings underlying the paired comparisons |
| [fusion_metrics.csv](details/fusion_metrics.csv) | Explicitly labeled endpoint and episode fusion |
| [calibration_matched_far.csv](details/calibration_matched_far.csv) | 500-split retrospective 5% FAR budget sensitivity |
| [expert_policy_score_distributions.csv](details/expert_policy_score_distributions.csv) | Episode-maximum score medians and interquartile ranges |
| [main_calibration_sensitivity_500.csv](details/main_calibration_sensitivity_500.csv) | All 500 calibration splits, including pooled summaries |
| [failure_categories.csv](details/failure_categories.csv) | Detection counts by task and failure category |

The public tables retain exact measurements. Blank values are unavailable.
Current seed IDs identify the four runs used in the paper; archived and
development runs are excluded. The protocol records the distinction between
calibrated thresholds, retrospective thresholds, and conditional online latency.
