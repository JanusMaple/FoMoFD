# Project website

The included static website provides simulation and real-robot replay panels,
detection results, sensitivity experiments, and latency comparisons.

[Open the interactive project website](https://anonymous.4open.science/w/FoMoFD/).

For a local preview, open the repository's `index.html` in a browser.

The page works directly from disk. It starts with Needle Drop and Shunt
Insertion Misalignment in wrist view, paused at 2x playback speed. Successful
calibration curves are visible by default. Switch tasks or cameras using the
panel controls, and drag a score curve or timeline to seek.

## Website files

`index.html`, `app.css`, `app.js`, and `replay.js` in this directory are the
presentation sources. The generated homepage is the repository-root `index.html`;
its styles, scripts, score data, downloads, and replay media live under `website/`.
Keep `index.html` and `website/` together when copying or hosting the page.
It uses no remote scripts or fonts.

The default exporter preserves this layout. A custom `--output` directory
contains a standalone website with its own `index.html`.

## Rebuild from experiment outputs

Rebuilding requires the recordings, cached scores, and benchmark reports
referenced by `lerobot/configs/benchmark/results_catalog.yaml`. These research
assets are not bundled; the included static website is already usable.

With those inputs available, run in the `lerobot` environment:

```bash
python lerobot/utils/export_benchmark_tables.py
python lerobot/utils/export_review_site.py
python lerobot/utils/export_public_results.py
```

To apply presentation changes while reusing exported videos and score curves:

```bash
python lerobot/utils/export_review_site.py --refresh-presentation
```

The exporter checks the benchmark inputs and cached-media hashes. Presentation
refresh also requires the source benchmark inputs. Evaluation definitions are
documented in [protocol.md](protocol.md).
