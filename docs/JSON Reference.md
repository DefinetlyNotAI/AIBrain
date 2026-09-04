# JSON reference

Analysis files are UTF-8 JSON, optionally wrapped in gzip when the filename ends in `.json.gz`. Current normal-chat
exports use `aibrain.session-analysis.v2`; Infinite-mode Analysis+ exports use
`aibrain.infinite-analysis-plus.v2`. Both schemas describe real-time inference telemetry.

Files are deliberately compact. They do not contain renderer arrays, per-node frames, model weights, source prompts,
hidden states, attention matrices, MLP activations, or chain-of-thought.

| Field | Meaning | Type / units | Classification |
|---|---|---|---|
| `schema` | Export contract identifier. | string | status |
| `created_at` | Export creation time. | ISO-8601 UTC | measured |
| `conversation` | Visible chat or Infinite simulation turns. | array | measured |
| `graph.kind` | `real_time_inference_telemetry_map`. | string | status |
| `graph.topology` | Always `display_only`; nodes and links are not model topology. | string | status |
| `graph.nodes`, `graph.edges` | Size of the rendered display layout. | counts | display-only |
| `graph.measurement_channels` | Ordered names of the nine normalized telemetry inputs. | string array | status |
| `graph.cluster_colours` | Stable display colour for each channel. | hex colour map | display-only |
| `measurement_integrity` | Human-readable measurement source and graph limitation. | string | status |
| `integrity` | Analysis+ statement identifying autoencoder inputs and excluded model internals. | string | status |
| `recorded_frame_summary` | Compact findings over the retained telemetry frames. | object | derived |
| `frames` | Number of retained telemetry frames included in the result. | count | measured |
| `mean_novelty`, `peak_novelty` | Autoencoder embedding-distance/reconstruction findings. | unitless | derived |
| `mean_reconstruction_error`, `mean_coherence` | Autoencoder reconstruction quality for this export. | unitless | derived |
| `most_active_channel` | Channel with the highest mean normalized measurement across retained frames. | channel name | derived |
| `maturity` | Persisted training-health state, conditions, and readiness. | object | health/status |
| `smart_analysis` | Present only in `aibrain.infinite-analysis-plus.v2`. | object | derived |
| `analysis_storage` | Paged/resident counts, cache limit and use, and data-loss status. | object | measured/status |
| `smart_analysis.neural_network` | Autoencoder architecture, feature schema, calibration, and lifetime health. | object | health/status |
| `session_findings` | Compact channel and temporal-pattern findings. | object | derived |
| `channel_profile` | Mean and peak normalized value for each measurement channel. | ratio 0–1 | normalized measurement |
| `key_events` | Up to 24 notable telemetry frames with raw metrics and learned findings. | object array | mixed |

Each key event identifies its generated step, visible `output_text` chunk, dominant channel, `Real-time` source, raw telemetry,
embedding novelty, reconstruction error, coherence, and compact embedding. Raw telemetry uses these fields when a live
logits snapshot is available:

| Telemetry field | Meaning | Units |
|---|---|---|
| `context_tokens`, `context_limit` | Evaluated context position and requested context size. | tokens |
| `output_tokens`, `chunk_tokens` | Retokenized visible count in this reply and current chunk. | tokens |
| `stream_latency_ms` | First-chunk latency or observed time since the previous visible chunk. | milliseconds |
| `retokenized_tokens_per_second` | Current visible chunk's retokenized count divided by its latency. | tokens/second |
| `vocabulary_size` | Number of raw logits inspected. | entries |
| `raw_logits_available` | Whether the chunk carried a valid raw-logit snapshot. | `0` or `1` |
| `raw_logit_entropy_bits` | Shannon entropy of the raw-logit softmax. | bits |
| `raw_top_probability` | Largest raw-logit softmax probability. | ratio 0–1 |
| `raw_top_five_mass` | Sum of the five largest raw-logit softmax probabilities. | ratio 0–1 |
| `raw_confidence_margin` | Difference between the largest and second-largest raw probabilities. | ratio 0–1 |
| `recent_output_occurrences` | Matches for this emitted text chunk in the previous 32 visible chunks. | count |

The raw-logit fields describe the distribution before penalties, grammar, top-k/top-p/min-p/typical filtering,
temperature, and token selection. They are not final sampling probabilities.

`maturity.state` is `Baby`, `Teen`, `Adult`, or `Elder`. Baby requires at least 2,048 persisted frames and sustained
recent consistency before Teen. Adult additionally requires at least 32,768 frames and sustained consistency/learning
slowdown. Elder requires a sustained post-Adult overfitting signal and freezes training weights. This is not an
accuracy claim. Baby and Teen exports carry `readiness: "caution"`; Adult and Elder carry `readiness: "ready"`.

The NPZ model stores `feature_schema=aibrain.realtime-inference-telemetry.v1`. An older NPZ without that identifier may
contain retired simulated features and is rejected instead of mixing those weights with real-time telemetry. Generate a
new response to create a compatible model. The standalone inspector reports the reason explicitly.
