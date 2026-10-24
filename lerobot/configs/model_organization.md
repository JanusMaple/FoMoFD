# Model locations and names

Store trained models under `fomofd/tasks/<task>/model/<family>/<run>/`.
Benchmark result folders contain reports, configuration snapshots and score
provenance, not checkpoint weights.

FM-WM runs use:

```text
fm_wm_vfmfvae_z8_both[_concat_channels]_state_off_k<K>_l<L>_action_<MODE>_data<SIZE>_seed<SEED>
```

For example:
`fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed1`.
`concat_channels` identifies the single fused-view backbone. Its absence
identifies the original separate branches. K is the endpoint horizon; L is the
history length. Action modes are `both`, `adaln`, `cross_attention`, and `none`.
Data labels 33/66/100 mean one-third/two-thirds/all of the expert training split.
The seed is the recorded training seed.

Legacy mappings were determined from saved configs, not folder suffixes:

| Old suffix | New field |
| --- | --- |
| H1 / H2 / H3 | k1 / k2 / k3 |
| AM | action_adaln |
| AT | action_cross_attention |
| NA | action_none |

The old ring action variants were K3; the replacements are K4. They are separate
runs and both remain available under their true K values.

VAE-observation baselines use names such as
`logpzo_vfmfvae_z8_fix_state_on_data100_seed0` and
`rnd_vfmfvae_z8_wrist_state_off_data100_seed0`. Historical shunt models trained
on PRL data have `prl_legacy` instead of `data100`. Policy-native baseline names
retain `act_tokens_q16_d768`. ACT checkpoint-step folders and VAE latent-size
names retain their separate conventions.

Each run contains `checkpoints/best.pt`, `checkpoints/last.pt`, saved training
configuration and, where available, `history.jsonl` and `splits.json`.
Imported runs retain their original `config.json`. A `config.local.json` contains
the previously validated desktop path overrides; the unified benchmark prefers
it and passes its complete merge to the scorer. Original metadata remains
byte-for-byte preserved in the run and `provenance/benchmark_adapter/`.

The optional, generated `fomofd/tasks/model_locations.json` records path-only
relocations. Current configurations use canonical paths. Benchmark utilities
resolve historical paths through this map and continue checking the original
hashes; immutable score caches and archived reports are not rewritten. Cached
scores can be reused after a path-only move. Changes to model/scorer bytes still
invalidate ordinary cache matching; `--cache-only --reuse-score-report` explicitly
recomputes metrics from an existing report's verified cached scores.
