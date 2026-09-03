# JSON reference

Analysis files are UTF-8 JSON, optionally wrapped in gzip when the filename ends in `.json.gz`. They are deliberately
compact: no raw renderer frames, model weights, source prompts, or transformer activations are exported.

| Field | Meaning | Type / units | Classification |
|---|---|---|---|
| `schema` | Export contract identifier. | string | health/status |
| `created_at` | Export creation time. | ISO-8601 UTC | measured |
| `conversation` | Visible chat or simulation turns. | array | measured |
| `graph.nodes`, `edges`, `regions` | Generated visual-connectome topology summary. | counts/names | simulated |
| `recorded_frame_summary` | Summary of recorded visual frames. | object | derived |
| `frames`, `mean_novelty`, `peak_novelty` | Recorded-frame count and embedding-distance findings. | count/unitless | derived |
| `mean_reconstruction_error`, `mean_coherence` | Autoencoder reconstruction quality for this export. | unitless | derived |
| `maturity` | Persisted training-health state, conditions, and readiness. | object | health/status |
| `smart_analysis` | Present only in `aibrain.infinite-analysis-plus.v1`. | object | derived |
| `analysis_storage` | Paged/resident record counts, cache limit, cache bytes, and whether oldest pages were discarded. | object | measured |
| `smart_analysis.neural_network` | Autoencoder architecture, calibration and lifetime health. | object | health/status |
| `session_findings` | Compact regional and pattern findings for this export. | object | derived |
| `regional_profile` | Per-region mean and peak visual activity. | unitless | simulated/derived |
| `key_events` | Up to 24 notable visual-frame events. | object array | derived |

`maturity.state` is `Baby`, `Teen`, `Adult`, or `Elder`. Baby requires at least 2,048 persisted frames and sustained
recent consistency before Teen. Adult additionally requires at least 32,768 frames and sustained consistency/learning
slowdown. Elder requires a sustained post-Adult overfitting signal and freezes training weights. This is not an accuracy
claim. Baby and Teen exports carry `readiness: "caution"`; Adult and Elder carry `readiness: "ready"`.

Older NPZ files have no rolling metric history. Their learned tensors and frame count are retained, but they are
conservatively inspected as Baby until fresh metrics are recorded. Missing history never grants Adult/Elder readiness.
