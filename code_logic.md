# LuminRAG execution and data flow

## Durable browser, temporary compute

The React application stores each knowledge base, its original files and chat in Dexie/IndexedDB. A canonical random UUID identifies each workspace and acts as a bearer capability. Keep it private. The backend never stores a durable hosted user database. It caches SQLite chunks, normalized NumPy vectors, NetworkX graph data and derived images in a per-workspace temporary directory.

A single fetch helper (`frontend/src/utils/api.ts`) supplies workspace and request credentials. A missing server cache returns409/workspace_not_loaded; the client hydrates one saved bundle and retries once. Concurrent requests share one hydration promise per workspace. Hosted cache eviction is bounded and idle entries expire after about30minutes. Active requests and jobs pin their cache.

`.lumin` files are ZIP archives. The compute bundle contains manifest.json, chunks.jsonl, embeddings.npy, embedding_ids.json, graph.json and assets/. Browser exports additionally contain browser.json and originals/. Originals and conversations stay in the browser during server hydration; compute needs only chunks, graph, vectors and derived assets. Provider keys and workspace capabilities are excluded from exports. Different embedding models trigger re-embedding.

## Credentials and models

`backend/credentials.py` parses X-LLM-Provider, X-LLM-Key, X-LLM-Model-Fast and X-LLM-Model-Smart. It deep-copies configuration. `backend/llm.py` is the only text/vision/structured provider transport. It uses the Anthropic and OpenAI SDKs and Instructor/Pydantic validation. Hosted endpoints fix provider base URLs and require explicit credentials; self-hosting can use .env/config and Ollama.

Fast models handle extraction, routing, captioning, decomposition, reflection. Smart models compose answers. Fast query mode skips question decomposition and reflection gates; Thorough retains the full verification pipeline. Retrieval uses FastEmbed BAAI/bge-small-en-v1.5 passages/query encoding, NumPy exact cosine top-k, and Xenova/ms-marco-MiniLM-L-6-v2 ONNX reranking.

Direct OpenAI, Anthropic and OpenRouter use fixed official endpoints; native
OpenAI uses strict structured output and completion-token budgets. Model
preferences persist, while all customer keys stay in memory. Skipped and
unavailable verification have explicit states and neutral UI; Thorough mode
withholds an answer when its source-support check fails.

Final answer generation returns Markdown and numbered supporting quotes in one structured provider call. Quote spans must match the actual numbered context passage and a citation used in the answer. Unverifiable quotes are discarded. Evidence responses retain complete passages, original filenames, sections, PDF page ranges and slide numbers. The browser lazily opens original-document previews, selects the cited PDF page or matching Word paragraph, and highlights validated supporting text. Missing originals and unmatched text retain a readable extracted-passage fallback. Word preview pages honor explicit breaks but do not promise Microsoft Word's pagination.

## Ingestion and publishing

Uploads are bounded, validated and staged. PDF and slide processors preserve text/page provenance and extract bounded source images. Image uploads use the shared captioner. Markdown/text use heading-aware chunking. Microsoft's MarkItDown handles local Office/HTML/EPUB streams with plugins disabled. Hosted mode rejects audio/video; self-hosted installations can install the optional A/V requirements.

The hierarchy pipeline extracts per-document topics, then groups up to eight
same-source passages and 6,000 source characters into one combined subtopic/content
request. Original citation passages remain separate. Each response accounts for
every passage, validates source IDs and parents, and checks excerpts against its
own cited passages. Up to four context-preserving I/O workers run independent
requests; immutable synonym pools avoid re-embedding existing names per proposal.
`graph_extraction.mode: separate` retains the earlier per-passage stages. The
hierarchy retains directed HAS_SUBTOPIC and HAS_CONTENT edges. A bounded
similarity shortlist proposes semantic pairs (up to16 neighbors,80 pairs,
batches of10). The verifier receives concept summaries and up to two source
excerpts per endpoint, each bounded to1200 characters. It returns a
related/not-related decision, finite confidence and supporting passage IDs.
Accepted IDs must come from the supplied context and cover both concepts.
Missing context or unsupported decisions produce no connection. This preserves
the existing linking-call pattern and numerical thresholds; it does not force
links between unrelated topics.

backend/utils/associations.py normalizes semantic pairs into a single symmetric RELATED_TO edge with a stable hashed identity and no display label. Separate automatic, manual, note-wikilink and legacy records retain their confidence and source IDs. Verified automatic backing is invalidated when any required passage is removed; manual backing survives. Source cleanup retains endpoints of surviving connections and their hierarchy ancestors, even if those concepts no longer have a source passage. Removing a wikilink only removes its owner's backing; explicitly deleting a connection removes all backing, which its inverse restores. Editing note citation text invalidates incident automatic grounding. Loading/exporting/importing old graphs folds reversed/labeled edges without model calls or upgrading legacy provenance to verified. Nodes, hierarchy, evidence and positions are preserved.

All state is prepared privately and published under the selected workspace's mutation lock. Self-hosted state keeps journal-backed crash recovery and source replacement. Ingestion job registries, versions, epochs and graph embedding caches are scoped to the workspace.

## Graph editing and retrieval

Sigma4/React Sigma6 render a Graphology multigraph: hierarchy edges have direction, semantic connections use plain lines. ForceAtlas2 runs in a worker and node coordinates persist. Evidence leaves remain chips rather than rendered nodes. Shared graph styles combine path, neighbor, selection and collapse state. Plain dragging moves nodes; Shift-drag connects; Ctrl-drag draws a lasso; the context menu exposes editing. Graph and Hybrid prompts treat connections as navigation hints and require cited passages for factual claims.

POST/api/graph/ops applies an ordered atomic batch to a candidate graph. Sparse inverse operations power undo/redo through Zustand/zundo. Notes have Markdown bodies and [[wikilinks]], create note chunks for dense retrieval, and expose citations. Editing semantic fields invalidates their graph embedding signatures. Merging rewires edges, unions evidence and terms, and updates parent references. Duplicate suggestions compare same-type embeddings in bounded blocks.

## Deployment boundary

The default Docker Compose stack runs the website, slim hosted backend and a
dedicated Cloudflare Tunnel connector in one isolated network namespace, without
host ports. Only the connector receives the tunnel token. The backend has
bounded CPU, memory, temporary storage, workspace and inference admission limits.
The hosted image is also portable to other supported container hosts. Readiness
is public and contains no corpus information. ALLOWED_ORIGINS is an exact
frontend-origin allowlist. Rate/body/file/archive/image/model-admission limits
protect shared compute. Self-hosted access retains its local/token authentication.
