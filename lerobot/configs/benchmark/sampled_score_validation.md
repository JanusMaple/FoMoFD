ACE and WM-PU normal-data validation
==================================

ACE is an **ACT-based adaptation of FIPER's action branch**, not full FIPER.
WM-PU is a **Ward-inspired uncertainty score on FoMo-FD's FM-WM**, not a
reproduction of Ward's world model. Neither score is reversed to improve AUROC.

ACE uses fixed XYZ histogram widths fitted to normal conditional sample
spread. Expert fitting/checking and production batching checks cover all four
tasks. This bin-resolution rule adapts the original workspace-bin construction;
resolving numerical variation alone does not establish useful failure uncertainty.

WM-PU uses one RK4 step per sample, with 16 samples and 64 vector-field
evaluations per score. This is a compute-budget choice, not a numerical
convergence claim. The multistep checks below are optional diagnostics; their
settings do not replace the reported one-step configuration.

`sampled_score_validation.yaml` declares all selection rules before evaluation:

1. Save exact expert-validation episode IDs, checkpoints, normalization and
   feature provenance. Keep policy calibration/test episodes out of fitting.
2. ACE: use five equally spaced frames per expert episode and 32 ACT prior
   samples. Split expert episodes deterministically into 60% fitting and 40%
   checking. The reference XYZ spread is the median within-frame/sample IQR.
   Select the coarsest fit-passing fixed width from the declared multiplier
   grid. Require the same widths to pass the disjoint checking episodes.
   Degeneracy, saturation and entropy-range criteria are numerical health
   checks, not a failure-detection objective. A failed check rejects the candidate settings.
3. WM-PU: use three randomly selected expert-validation episodes, three eligible
   windows per episode (10%, 50%, 90%), and 16 identical Gaussian initial samples
   at every RK4 refinement. Try 1, 2, 4, 8, 16, 32, 64 steps. For both cameras,
   require median relative SD error ≤1%, 95th percentile SD error ≤5%, and
   95th percentile endpoint NRMSE ≤5% on two successive refinements. Select the
   coarsest passing count. Reaching the cap is not a convergence certificate.
   These checks establish convergence only on the sampled normal windows.
4. Freeze the validated settings, retaining hashes and all candidate results.
   Generate new expert score trajectories and freeze functional templates for
   the changed score definitions. Confirm ACE's fixed widths pass the same
   normal holdout under production CUDA batching before any failure evaluation.
5. Rescore and recalibrate on the original 19 successful policy episodes,
   alpha=0.05, using strict `>` crossing. Evaluate unchanged test IDs. Preserve
   original caches and reports; revised cache identity includes the manifest,
   helper code, histogram widths and integration settings.
6. Report RK4 steps and vector-field evaluations: WM-PU with S=16 and N steps
   requires 16 solves and 64N field evaluations per window (both views together).
   New offline results do not replace online latency measured with old settings.

Run the numerical diagnostics with:

```bash
python lerobot/utils/validate_sampled_scores.py prepare --output NEW_RESULT_FOLDER
python lerobot/utils/validate_sampled_scores.py ace --output NEW_RESULT_FOLDER --device cpu
python lerobot/utils/validate_sampled_scores.py wm --output NEW_RESULT_FOLDER --device cuda
python lerobot/utils/validate_sampled_scores.py freeze --output NEW_RESULT_FOLDER
```

The benchmark YAML sets `ace.sampling_manifest` and `wm_pu.sampling_manifest`
to `NEW_RESULT_FOLDER/frozen_sampling.json`. Original settings remain the
default when these fields are absent. A manifest is checkpoint-specific.
Templates must be regenerated for the revised score definitions; an old
functional template cannot be reused with the new scores.

`sampled_score_diagnostics.py` makes CPU-only CSV/JSON/Markdown/HTML reports and
plots from saved scores. By default it examines only calibration successes.
It compares score maxima with window count, relative-phase score averages and
nested-prefix maxima. Relative phase uses the final duration and is strictly a
descriptive offline variable. Failure-group diagnostics require matching frozen
revised scores and cannot select settings. Length/phase associations are not a
complete explanation of below-chance AUROC.
