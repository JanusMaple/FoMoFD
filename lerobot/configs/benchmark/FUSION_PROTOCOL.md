# Endpoint and episode alarm fusion

Supplemental analysis: `fusion_aggregation_levels_v1`. Use the current frozen reporting protocol: v3 retains the data mask, unsmoothed scores and component thresholds, and uses the mean annotated wall-clock onset per view. Both aggregation levels use that same reference. Existing endpoint results are preserved and verified against the reconstruction; this analysis does not replace the primary detectors or select a fusion rule from failure performance.

Components are the separate-branch fixed/wrist WM-NC detectors, or WM-NC plus native-feature logpZO **within one view**. The concatenated-view trained world model is a separate architecture experiment.

Let `a_t` and `b_t` be component scores normalized by their saved calibration thresholds and positive scales: `(score − threshold) / scale`. Both rules use the **same intersection of eligible raw endpoints**, with matching wall timestamps. Native-only endpoints outside WM support remain excluded, so this comparison isolates aggregation order.

| Rule | Episode score Q | Positive decision |
|---|---|---|
| Endpoint OR | `max_t max(a_t, b_t)` | Either component alarms at an eligible endpoint. |
| Episode OR | `max(max_t a_t, max_t b_t)` | Either component alarms within the episode. Identical to endpoint OR on this support. |
| Endpoint AND | `max_t min(a_t, b_t)` | Both components alarm at the same endpoint. |
| Episode AND | `min(max_t a_t, max_t b_t)` | Each component alarms within the episode; the endpoints may differ. |

Use strict `Q > 0`, without fitting a new fusion threshold. For failure FDR, restrict **both** component streams to endpoints at/after the output-view onset: the earlier camera onset for view fusion, or that view's onset for within-view method fusion. Thus both alarms required by episode AND must occur within this eligible post-onset interval. A component's pre-onset alarm alone cannot satisfy the AND rule. Missing onset annotations remain unknown.

Successful-test FAR and episode AUROC use the whole valid episode. AUROC averages tasks equally. The separately labeled matched-FAR analysis selects thresholds from successful test maxima and uses post-onset failure maxima. It is retrospective and does not alter the saved component calibration. OR does not inherit an unchanged 5% joint false-alarm guarantee.

Episode AND is a retrospective agreement statistic, not simultaneous confirmation or a newly evaluated online monitor. No online latency is assigned to it or inferred from the historical endpoint replay. A causal latched-AND deployment would need its own explicit policy and evaluation.
