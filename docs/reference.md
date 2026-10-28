[← Back to LuminRAG](../README.md)

# Configuration and API

| Variable | Purpose |
|---|---|
| `LUMIN_HOSTED=1` | Temporary isolated workspaces, per-request BYOK, no environment provider keys or A/V |
| `ALLOWED_ORIGINS` | Exact allowed frontend origins, comma-separated; wildcard is rejected |
| `HOST`, `PORT` | Listener address/port; hosted defaults 0.0.0.0:8000 |
| `LUMIN_MAX_WORKSPACES` | Bounded workspace cache; default8 |
| `LUMIN_MAX_WORKSPACE_BYTES` | Temporary workspace disk budget; default268435456 (2 5 6 MiB) |
| `FASTEMBED_CACHE_PATH` | Shared ONNX model cache; baked into the hosted image |
| `LUMIN_MODEL_THREADS` | ONNX threads per model; default2 |
| `LUMINRAG_INGEST_LLM_WORKERS` | Graph-extraction provider I/O ceiling; default8, hard maximum8; independent of CPU threads |
| `LUMINRAG_MAX_UPLOAD_BYTES` | Per-file byte limit |
| `LUMINRAG_MAX_CONCURRENT_INFERENCE` | Process-wide answer-generation admission bound |
| `FORWARDED_ALLOW_IPS` | Optional exact trusted proxy IPs/CIDRs for client-IP limits; never set blindly to `*` |
| `LUMINRAG_API_TOKEN` | Remote self-host access only |
| `LUMINRAG_ENABLE_DOCS=1` | Explicitly enable API documentation, otherwise disabled in hosted/token-protected mode |
| `LUMINRAG_DATA_DIR` | Persistent self-host data directory only |

## API and implementation

`backend/utils/` and `frontend/src/utils/` contain shared helpers. See [architecture](../code_logic.md) for data flow and [contributing](../CONTRIBUTING.md) for development guidelines.

The central compute transport is backend/llm.py. FastEmbed supplies bge-small-en-v1.5 passages/query embeddings and the MiniLM ONNX cross-encoder reranker. Dense retrieval reranks its vector candidates. Graph retrieval searches graph nodes and expands connected concepts; hybrid combines graph and dense evidence. Both answer from retrieved source passages. The UI highlights the returned graph nodes. NumPy performs exact cosine search; NetworkX stores a directed multigraph, with one canonical edge per unordered semantic pair. MarkItDown converts Office/HTML/EPUB streams. The frontend uses React19, Sigma4, React Sigma6, Graphology, Dexie, fflate, Zustand and zundo.

Answer generation returns Markdown and supporting quotes in the same provider call. Each quote is checked against its numbered source passage before it can become a highlight. Invalid or missing quotes fall back to the source passage. PDF page offsets and slide metadata carry the location through retrieval, saved chat and document previews. React-PDF/PDF.js and docx-preview render originals lazily on the device. Word previews use a script-disabled frame, restricted resource policy and bounded archive validation.

Fast answers carry `verification_status: skipped`; failed verification calls carry
`unavailable`. Neither displays positive verification badges. Thorough mode
withholds an answer when its source check reports unsupported factual claims.
An AI review can still miss errors; exact quote matching verifies a quote's
location, not the truth of every statement in the answer.

Explicit multi-document comparisons and summaries use a separate 12,000-character
passage budget (`generator.multi_document_max_context_chars`); ordinary questions
keep 4,000 (`generator.max_context_chars`). The two settings are independent and
each configured value is used as is. Per-question source caps still apply, and
generation and answer verification see the same bounded numbered passages. A
larger budget does not guarantee answer quality or latency, and it adds no
provider calls and changes no provider or model.

`graph_extraction` in `config/llm.yaml` defaults to compact document extraction:
up to 64 same-source passages and 32,000 source characters per request, with
topics, subtopics and concise concept cards produced together. The original
retrieval passages and page/slide references stay unchanged. A slide and its
speaker notes form one extraction view, retaining every original passage ID.
Every view needs an explicit evidence or no-content classification. Every card
requires a primary source; unrecognized extra references are discarded only
when valid independent backing still covers the view. Uncovered material,
incomplete output and invalid primary references fail before publication.
Compact cards omit repeated quotations and keyword lists; answers still return
quotes checked against the full original passages.
`mode: combined` retains the smaller eight-passage extraction path, and
`mode: separate` retains the earlier per-passage stages.

Private local Ollama uses `local_mode: combined` by default, keeping extraction
requests within eight passages, 6,000 source characters and four workers. This selects the
existing smaller extraction path without changing the provider or model.
To opt into larger requests, set `local_mode: compact` and configure at least
a 32K context on the Ollama server, for example with
`OLLAMA_CONTEXT_LENGTH=32768` or a model created from a Modelfile containing
`PARAMETER num_ctx 32768`. Confirm the allocated context with `ollama ps` and
allow additional room for unusually token-dense sources. Larger contexts need
more memory. Ollama's OpenAI-compatible API cannot set `num_ctx` per request.
See the official [context guide](https://docs.ollama.com/context-length) and
[model configuration instructions](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size).
Remote Anthropic, OpenAI, OpenRouter and Ollama Cloud keep compact extraction.

Source indexing overlaps provider requests. Extraction and connection
verification use at most eight concurrent provider requests by default, while
native embedding work remains serialized in bounded batches. Connection checks
retain their existing candidate and evidence limits. Valid graph search vectors
survive candidate/publication cloning, so the first question does not rebuild
the same node embeddings. Publication of passages, graph and vectors remains
atomic; no unfinished graph is presented as complete. Processing time depends
on document size, CPU availability, provider latency and server queueing.

New Anthropic selections default to `claude-haiku-5-5` for extraction and
`claude-sonnet-5-5` for answers. Saved model preferences remain editable; no
visitor credential is stored with them.

Hosted requests carry `X-Workspace-Id` (canonical random UUID4). Provider operations additionally use `X-LLM-Provider`, `X-LLM-Key`, `X-LLM-Model-Fast` and `X-LLM-Model-Smart`. The UUID is a secret capability, so keep it out of URLs and logs. There is no application login in v1.

`X-LLM-Provider: ollama_cloud` selects Ollama Cloud with the same request-scoped
key and model headers. Its endpoint is fixed to `https://ollama.com/v1`; local
`ollama` remains self-hosted only. Cloud uses `max_tokens` and requests
`reasoning_effort: none`, which model-specific controls may interpret differently.
All Cloud structured requests use prompt JSON followed by Pydantic validation,
including when configuration previously selected a native-schema mode. No native
`response_format` or tool requirements are sent. Cloud HTTP requests cannot
inherit environment or server-configuration keys when request credentials are
absent, and no shared Cloud key belongs in the deployment environment.
See [Ollama's compatibility documentation](https://docs.ollama.com/api/openai-compatibility)
and [Cloud structured-output limitation](https://docs.ollama.com/capabilities/structured-outputs).

- `GET /api/health`, `GET /api/ready`: public capabilities/readiness.
- `GET /api/workspace/status`: `{loaded,version}`.
- `POST /api/workspace/import`: multipart `.lumin` file, or `empty=true` to initialize.
- `GET /api/workspace/export`: portable compute bundle.
- `POST /api/graph/ops`: atomic ordered operations with sparse inverses.
- `GET /api/graph/duplicates`: same-type semantic merge suggestions.
- `DELETE /api/data`: clears only the selected knowledge base.

Unloaded caches return 409 with `code: workspace_not_loaded`. Hosted mode rejects legacy source/S3/global static routes. Imports reject unsafe paths, object arrays, incompatible NumPy headers, excessive member sizes and host-file references. SQL is parameterized; keys are omitted from logs/errors/archives. Request/file/model/graph/image limits bound shared compute. Public service still needs monitoring and appropriate capacity settings for real traffic.

Hosted limits use both the connection's peer and its workspace: 20 compute requests and five uploads per minute, with three new workspace imports per minute. Known workspace IDs can rehydrate within the upload allowance. Incoming bodies have a 15-second idle deadline and a five-minute total deadline; processing continues under its separate limits after the body completes. Configure only the exact trusted proxy addresses when deploying behind an ingress, since visitors sharing one peer address also share its allowance.

Office, PDF and slide parsing run in credential-free subprocesses with 60-second and 5 1 2 MiB limits, bounded output and cleanup. Spreadsheets additionally limit logical cells, including sparse gaps. Optional audio/video processing has duration, frame, decoded-byte and subprocess limits. Linux workers apply hard address-space/CPU/file limits; Windows uses the process-tree memory/time watchdog. These controls bound resource use; they are not a separate operating-system security sandbox against an unknown native-parser exploit.
