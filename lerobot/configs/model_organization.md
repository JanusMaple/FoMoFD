# Model locations

Store models under `fomofd/tasks/<task>/model/<family>/<run>/`.
The model families include `act_dinov2`, `vfmf_vae`, `fm_wm`,
`faildetect_logpzo`, and `fiper_rnd`.

World-model run names describe the representation, views, state input,
prediction horizon K, history length L, action conditioning, training-data
fraction, and random seed:

```text
fm_wm_vfmfvae_z8_both[_concat_channels]_state_off_k<K>_l<L>_action_<MODE>_data<SIZE>_seed<SEED>
```

For example, the main model is
`fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed0`.
`concat_channels` identifies the fused-view backbone; otherwise the model has
separate camera branches. Action modes are `both`, `adaln`, `cross_attention`,
and `none`. Data labels `33`, `66`, and `100` mean one third, two thirds, and all
of the expert training split.

A world-model run contains its effective `config.json`, episode split
`splits.json`, training history, and `checkpoints/best.pt` and
`checkpoints/last.pt`. Scoring reads the effective configuration beside the
checkpoint. Keep those files together when moving a trained model.

ACT checkpoints use LeRobot's `pretrained_model/` and `training_state/` layout.
Policy-feature baselines include `act_tokens_q16_d768` in their run names;
supplementary VAE-feature baselines use `vfmfvae_z8`.
