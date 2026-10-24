# Qwen video baseline

The paper baseline is Qwen3-VL-8B-Instruct with the v9 prompt, independent
fixed/wrist inputs, the full observed video history sampled at 1 FPS, and a
256-token generation cap. Inputs never include future frames. Binary decisions
have no conformal threshold or continuous-score AUROC.

Configuration: [protocol_video_v9_qwen.yaml](protocol_video_v9_qwen.yaml).
Prompt: [prompt_video_v9.txt](prompt_video_v9.txt).
Environment: [VLM setup](../../../environments/README.md).

The full-test evaluation covers 400 test recordings in each camera. It excludes
calibration recordings and does not use another detector's skip mask. FDR in the
paper includes late returned detections; paired latency uses on-time detections.
Current summaries interpret the saved traces using the three-annotation mean
onset in [the v3 evaluation protocol](../evaluation_protocol_v3.yaml).

The benchmark supports separate preparation and inference commands:

```bash
python lerobot/utils/vlm_online_benchmark.py --help
```

Choose a new output directory for a changed protocol. Resume uses recorded
outcomes; interrupted or indeterminate attempts are not silently retried.
Per-recording outputs preserve prompts, responses, frame timestamps, decisions,
and timing so that the public aggregate can be reproduced.

Earlier prompt and protocol configurations are retained for source
reproducibility. Their development results are excluded from public downloads
and are not substituted for the paper's v9 baseline.
