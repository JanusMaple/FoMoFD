# AEU: Action Ensemble Uncertainty

Source: [Thompson et al., Early Failure Detection in Autonomous Surgical Soft-Tissue Manipulation via Uncertainty Quantification](https://arxiv.org/html/2501.10561v2).

**Report names:** **AEU** is our primary positional variant (signed change in positional ensemble variance); **AEU-Rot** is the supplementary rotational variant. AEU is our descriptive abbreviation for Action Ensemble Uncertainty, not a method name claimed by the source paper. Both variants use both cameras. Internal IDs remain `act_ensemble_pos` and `act_ensemble_rot`, preserving cache and checkpoint identities.

Main-method charts and tables show AEU only. AEU-Rot remains available under **All comparisons**, the **Supplementary detectors** latency filter, and the supplementary summary table; original score records are retained.

This is an ACT-based adaptation, not a DeformerNet reproduction. The paper uses five independently initialized goal-conditioned point-cloud policies and primarily assesses the initial change in position/rotation disagreement. We use three existing RGB/state ACT policies (seeds 1000, 1001, 1002), all at the final 200,000-step checkpoint, on the original recorded single-policy rollouts. We do not execute an ensemble controller or add interventions.

Each member uses its own saved state/action normalization, learned visual token compressor and ACT weights. The shared DINO backbone is reused only after exact weight equality checks. Evaluation mode disables dropout/state augmentation. The VAE latent is zero, as in deterministic ACT deployment; this measures between-policy disagreement, unlike ACE's within-policy prior samples. The first predicted action of each four-action chunk is used, without temporal action aggregation.

For each observation, compute:

- Position: population variance `mean_i ||p_i - mean_j p_j||²`, in m².
- Rotation: project the matrix mean to SO(3) by SVD with determinant correction, then average the geodesic angle to that mean, in radians. This follows the paper's angular dispersion definition, despite its term “rotational variance.” Repository rotation-6D actions concatenate the first two matrix columns. Gripper values do not enter either score.
- Score: current dispersion minus dispersion at the immediately preceding prepared observation, separately for position and rotation. Higher signed differences are anomalous. There is no absolute value, clipping, time normalization, smoothing, or test-driven sign reversal. The first observation has no difference score.

The two companion detectors are `act_ensemble_pos` and `act_ensemble_rot`; both use both policy cameras. They are not fixed/wrist-specific detectors. Offline prediction uses batch 8 and shares a checksummed physical-action cache between the two scores. Online processes one eligible two-observation window at a time; a skipped previous observation is inferred on demand and charged to decision latency. The previous *processed* observation is never substituted for the previous *recorded* observation.

Current reporting uses [the frozen v3 protocol](evaluation_protocol_v3.yaml):
19 calibration successes, strict threshold crossing, valid endpoints, and
post-onset detection relative to the three-annotation mean. Both episode
thresholds and expert-derived functional bands are available. Online latency
uses decision completion and reports late returns separately.

Settings were selected before failure scoring. Development validation uses only the first eight observations of success-calibration episode 0 in each task. It checks finite physical outputs, rotation validity, nonzero member diversity, batch/single agreement, memory and throughput. No failure results select the checkpoint, action horizon, score sign, or batch size.

```bash
python lerobot/utils/unified_benchmark.py \
  --config lerobot/configs/benchmark/act_ensemble.yaml --dry-run

python lerobot/utils/unified_benchmark.py \
  --config lerobot/configs/benchmark/act_ensemble.yaml --allow-inference \
  --report-dir fomofd/tasks/benchmark_results/experiments/NEW_RUN/offline_act_ensemble_raw

python lerobot/utils/benchmark_report_protocol.py \
  --source fomofd/tasks/benchmark_results/experiments/NEW_RUN/offline_act_ensemble_raw \
  --output fomofd/tasks/benchmark_results/experiments/NEW_RUN/offline_act_ensemble
```

The export step verifies the saved configuration, onset snapshot, post-onset row
semantics and score hashes before attaching the frozen-protocol declaration
required by the online planner. It preserves numerical results and source reports.
Launch replay with this checkout's `lerobot`, `lerobot/utils` and
`lerobot/policies/act_dinov2/src` directories on `PYTHONPATH`, as the experiment
queue does; this avoids resolving an editable ACT installation from another repo.

The corrected continuous-score replay retains the existing offline-detected-failure selection, now based on the primary annotator's post-onset detection. It replays each selected episode through source exhaustion, recording every returned decision. This measures conditional online latency/retention; it does not measure full-test online FAR/FDR. Onsets never control scheduling or stopping. Source-end late returns are reported separately.

Current conditional replay includes the main baselines and K/L sweeps.
Action and training-data variants have offline results; no unmeasured online
latency is inferred from those scores.

The separate Qwen stage uses the currently selected main-report v9 prompt, model revision, 1-FPS full observed history, 256-token cap, separate cameras and existing parsing/decoding settings. Only full-episode replay and frozen onset interpretation change. It evaluates all 800 test episode/view combinations, with no offline skip or new continuous AUROC. Existing results and prompts remain intact.
