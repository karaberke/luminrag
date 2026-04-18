# LuminRAG

**Adaptive Self-Verifying Graph-Augmented QA for Education**

LuminRAG is a research prototype for question answering over multimodal university course content. It combines dense vector retrieval, graph-based multi-hop reasoning (GraphRAG), and Self-RAG-style self-reflection into a single adaptive pipeline — and surfaces the full reasoning trace through an interactive concept knowledge graph in the browser.

The system ingests lecture videos, slide decks, PDFs, audio recordings, and images; constructs a concept knowledge graph from the extracted entities and relations; and answers natural-language questions by routing each query through whichever retrieval strategy is most appropriate. Every answer is accompanied by source evidence, a routing label, and a four-axis Self-RAG verdict.

---

## Research hypothesis

Does augmenting a traditional dense RAG pipeline with a self-verifying, graph-based multi-hop retrieval system significantly improve factual grounding and answer accuracy on complex, university-level educational queries?

### Baselines

| System | Description |
|--------|-------------|
| Naive RAG | Vector DB + dense embeddings + LLM generation |
| Pure GraphRAG | Graph-based multi-hop retrieval only (no Self-RAG gate) |
| Pure Self-RAG | Vector retrieval + reflection gate (no knowledge graph) |
| **LuminRAG (Full)** | Multimodal ingestion + adaptive routing + graph/vector retrieval + Self-RAG verification |

### Benchmark datasets

- **HotpotQA** — multi-hop QA gold standard
- **MuSiQue** — rigorous multi-hop, resistant to single-hop shortcuts
- **ScienceQA** — multimodal science questions (text + images)
- **QASper** — QA over NLP research papers
- **Lumin-Edu (custom)** — questions from actual university lecture videos, slides, and PDFs

### Metrics

- **Hit Rate @ K** and **MRR** — retrieval quality
- **Faithfulness** (hallucination rate) — maps to Self-RAG `is_supported`
- **Answer Relevance** — maps to `is_useful`
- **Context Precision** — relevance of retrieved context

---

## Architecture

```
Multimodal Sources → [Ingestion] → [Document Store + Vector Index + Concept Graph]
                                                    ↓
              Query → [Multi-hop Decomposer] → per sub-question:
                                                    ├─ Query Router  →  "dense" | "graph" | "none"
                                                    ├─ Dense Retrieval (FAISS)
                                                    │   or Graph Retrieval (BFS over concept graph)
                                                    └─ Self-RAG gate (reflect_retrieval)
                                                    ↓
                                              [Generator + reflect_answer]
                                                    ↓
                                         Answer + Citations + Graph Hop Trace
```

### Storage

| Store | Engine | Purpose |
|-------|--------|---------|
| Document store | SQLite | Raw text chunks and source metadata |
| Vector index | FAISS `IndexFlatIP` | Dense retrieval; L2-normalised embeddings ≡ cosine similarity |
| Concept graph | NetworkX + JSON | Multi-hop traversal and browser mindmap rendering |

---

## Quick start — Docker

**Requirements:** Docker, an [Anthropic API key](https://console.anthropic.com).

```bash
git clone <repo>
cd luminrag

cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY=sk-ant-...

docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost |
| Backend API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

Ingested data (SQLite DB, FAISS index, concept graph) is persisted in a named Docker volume and survives restarts. `config/` is bind-mounted so LLM settings can be changed without rebuilding.

---

## Local development

**Requirements:** Python 3.11+, Node.js 20+, pnpm, ffmpeg.

```bash
# Python environment
python -m venv .venv
source .venv/Scripts/activate     # Windows bash
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
cp .env.example .env              # then fill in ANTHROPIC_API_KEY

# Backend — http://localhost:8000
python -m backend

# Frontend — http://localhost:5173 (separate terminal)
cd frontend
pnpm install
pnpm dev
```

The Vite dev server proxies `/api/*` to `localhost:8000`, so both servers can run side-by-side without CORS configuration.

### Running tests

```bash
pytest                            # all 358 tests
pytest backend/tests/test_generator.py          # single file
pytest backend/tests/test_generator.py::test_fn # single test
```

---

## Ingesting course content

**Via the UI:** click **Upload** in the top-right corner to add files through the browser. The backend ingests them and rebuilds the vector index and concept graph live.

**Via CLI (batch ingestion):**

```bash
# Place raw files in backend/data/raw/, then run:
python scripts/ingest.py

# Preview what would run without executing
python scripts/ingest.py --dry-run

# Mark specific PDFs as slide decks (captioned per page instead of text-chunked)
python scripts/ingest.py --slides lecture_slides.pdf week2.pdf

# Point at a different folder
python scripts/ingest.py path/to/folder
```

Supported formats: `.pdf`, `.mp4/.mkv/.mov/.avi`, `.mp3/.wav/.m4a/.ogg/.flac/.aac/.wma`, `.jpg/.jpeg/.png/.webp`.

---

## Pipeline stages

### Stage 1 — Multimodal ingestion

| Module | Description |
|--------|-------------|
| `video_transcriber.py` | Whisper transcription + OpenCV SSIM keyframe extraction + LLM captioning; chunks by timestamp interval |
| `slide_processor.py` | One chunk per PDF slide page, captioned via LLM |
| `pdf_processor.py` | Semantic chunking of textbook/notes by font-size heading boundaries |
| `audio_processor.py` | Whisper transcription of standalone audio files |
| `image_processor.py` | Single-image captioning via Ollama or Anthropic |

### Stage 2 — Concept graph construction

Two-step extraction: fast local NER (GLiNER or spaCy) identifies entities; a lightweight LLM call maps relations between them. Valid relation types: `PART_OF`, `PREREQUISITE`, `CAUSES`, `EXAMPLE_OF`, `EXPLAINS`, `BELONGS_TO_TOPIC`.

### Stage 3 — Query routing

Routes each (sub-)question to the appropriate retrieval strategy using heuristics with an optional LLM fallback. "What is X?" / "Define Y" → `dense`; "Why does…" / "Compare A and B" → `graph`; general knowledge → `none`.

### Stage 4 — Retrieval

**Dense path** (`vector_retriever.py`): query embedding → top-K FAISS lookup → ranked chunks.

**Graph path** (`graph_retriever.py`): embed query → cosine-match anchor nodes → relevance-guided BFS (expand neighbour only if cosine similarity ≥ threshold) → collect subgraph triples + source chunks.

### Stage 5 — Self-RAG reflection

Two gates:
- `reflect_retrieval()` — pre-generation; checks `is_relevant` and skips irrelevant context.
- `reflect_answer()` — post-generation; checks all four verdicts. Both gates default to safe pass-through on LLM failure.

### Stage 6 — Multi-hop reasoning

Decomposes complex questions into ≤4 sub-questions, runs retrieval + reflection per sub-question, then merges all evidence into a single `RetrievalResult` for the generator.

### Stage 7 — Generation

Selects a prompt template (`DENSE_RAG_PROMPT`, `GRAPH_RAG_PROMPT`, or `NO_RETRIEVAL_PROMPT`) based on routing mode, calls the configured LLM, and returns the answer with inline `[N]` citations.

---

## Using Ollama instead of Anthropic

All LLM-calling stages support local [Ollama](https://ollama.com) as an alternative provider. Switch any section in `config/llm.yaml` from `provider: anthropic` to `provider: ollama` and set the corresponding model. The config is bind-mounted in Docker — no rebuild needed.

```yaml
# config/llm.yaml — example: switch the generator to a local model
generator:
  provider: ollama
  ollama_model: llama3.2:3b
  ollama_base_url: http://host.docker.internal:11434  # inside Docker
  # ollama_base_url: http://localhost:11434           # local dev
```

---

## Configuration reference

| File | Purpose |
|------|---------|
| `config/llm.yaml` | Provider/model for every pipeline stage; Whisper settings; chunking parameters; retrieval thresholds |
| `config/db.yaml` | Paths for SQLite store, FAISS index, and graph JSON |
| `.env` | `ANTHROPIC_API_KEY`; optional `HF_TOKEN` for private HuggingFace models |

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness — reports index/graph/chunk-count status |
| `GET` | `/api/graph` | Full concept graph as `{nodes, edges}` |
| `POST` | `/api/query` | Run the QA pipeline; body: `{"query": "...", "routing_mode": "dense"\|"graph"\|null}` |
| `POST` | `/api/ingest` | Upload files (multipart) and run the full ingestion pipeline in-process |
| `DELETE` | `/api/data` | Wipe all stored data and reset in-memory state |

Interactive docs at `/docs` when the backend is running.
