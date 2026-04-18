### **LuminRAG: Adaptive Self‑Verifying Graph‑Augmented QA for Education**


### **Goal**

Build an educational QA system that:
- Use virtual environment
- Uses **multimodal course content** (lecture notes, slides, pdfs, word documents, videos, audios, images) as ground truth.
- Combines **vector retrieval (RAG)**, **graph‑based retrieval (GraphRAG)**, and **Self‑RAG‑style self‑reflection**.
- Supports **multi‑hop reasoning** and exposes the reasoning path to the learner as an **interactive concept mindmap**.
- Using multimodal course content to chunk into nodes and build a graph. Everytime user query, queries the graph it will demonstrate visually which node it started from and which nodes it hopped to and retrieved by highlighting it.


***

### **System architecture overview**

The system has **three main layers**:

1. **Data / Storage layer**
2. **Retrieval / reasoning pipeline**
3. **User interface / visualization**

**Three databases**:  
- **Vector database** (for dense retrieval)  
- **Graph database** (for course‑concept KG and multi‑hop traversal)  
- **Light document store** for source metadata.

***

## 1. Data / Storage layer

### 1.1 Datasets and sources

- University‑level course materials in:
  - **Lecture videos**
  - **Slide decks (PDF / PPT)** 
  - **Textbook PDFs / notes**
  - **Images (jpeg, pdf, etc)**
  - **Audio (jpeg, pdf, etc)**
- Final QA benchmark: (In progress)

### 1.2 Database setup

You need **two full databases + one lightweight store**:

1. **Vector database**  
   - Role: fast dense and sparse retrieval for single‑hop questions and baseline RAG.
   - Example engines: FAISS
   - Stores:
     - `chunk_id`, `text`, `source_id`, `page/slide`, `modality` (video, slide, textbook), `embedding`.

2. **Graph database**  
   - Role: concept graph for multi‑hop reasoning, relations, and mindmap rendering.
   - Example engines: lightweight file‑based (NetworkX + JSON) for research‑scale.
   - Stores nodes and edges:
     - **Nodes**: concepts, documents, entities, prerequisite relations.
       - Properties: `name`, `type`, `description`, `source_ids` (list of chunk IDs).
     - **Edges**: `PART_OF`, `PREREQUISITE`, `CAUSES`, `EXAMPLE_OF`, `EXPLAINS`, `BELONGS_TO_TOPIC`, etc.  
       These are “pedagogically typed relations.”

3. **Document / metadata store**  
   - Role: prune & traceability from graph nodes back to original text.
   - Can be:
     - a JSON file per document, or
     - a simple SQLite table: `doc_id`, `source_path`, `title`, `modality`, `raw_text_snippet`.

***

## 2. End‑to‑end pipeline (modules to code)

Structure the code into these **modules** (each can be one Python file or class):

### 2.0 Shared Data Schemas (Pydantic)
To ensure robust data passing between modules, the system relies on the following core data models:
- Chunk: id, text, source_id, modality, metadata (dict).
- GraphTriple: head (str), relation (str), tail (str), source_chunk_ids (list).
- RetrievalResult: chunks (List[Chunk]), subgraph (List[GraphTriple]), routing_mode (str).
- ReflectionVerdict: is_relevant (bool), is_supported (bool), reasoning (str).

### 2.1 Stage 1 – Multimodal ingestion

- **Input**:  
  - Lecture videos, slide decks, textbook PDFs.
- **Output**:  
  - Parsed text chunks + metadata; concepts/relations for the graph.

Sub‑modules:

- `video_transcriber.py`  
  - Extract Audio & Transcribe - Use OpenAI Whisper locally to create a transcript with timestamps. 
  - Extract Keyframes:Use OpenCV to calculate structural similarity (SSIM) or frame differencing. Only extract and caption a frame when a major visual change occurs (e.g., the professor changes the slide) and Use a cheap llm to describe the keyframes.
  - Combine & Chunk
  - Apply Semantic and Structural Chunking rather than arbitrary token limits:
    - Videos: Chunk by logical timestamp intervals that align with keyframe/slide changes.
    - Slides/PDFs: Chunk by individual slide, markdown header, or semantic section.
    - Include extensive metadata: chunk_id, source_id, start_time / end_time (for video), page_number, and modality.
- `slide_processor.py`  
  - For PDF slides: extract text + pass slide images to an image‑caption model to get rich captions.
  - Store each slide as one or more chunks.
- `pdf_processor.py`  
  - Extract textbook / notes text, split into semantically coherent chunks.
- `document_store.py`  
  - Write all chunks + metadata to the document store (JSON / SQLite).

### 2.2 Stage 2 – Concept‑graph construction

- **Input**:  
  - Text chunks from all sources.
- **Output**:  
  - A concept knowledge graph in the graph DB.

Sub‑modules:

- `entity_extractor.py`  
  - Use a hybrid extraction pipeline to maximize speed and minimize API costs:
    - Step 1 (Entities): Use a fast, local NLP model like GLiNER (Zero-shot NER) or spaCy but make it users choiceto identify key domain entities (Enzyme, ActiveSite, ReactionRate).
    - Step 2 (Relations): Pass the extracted entities and the text chunk to a highly cost-effective LLM (users choice) strictly to map the relationships between the pre-identified entities.
- `graph_builder.py`  
  - Connect to the graph DB.
  - For each triple:
    - Create/get concept node for `head` and `tail`.
    - Create edge with relation type and `source_chunk_id` on edge.
  - Optionally:
    - Add `BELONGS_TO_TOPIC` edges to coarse topics (e.g., “Kinetics”, “Thermodynamics”).
    - Add `PREREQUISITE` edges between topics (e.g., “Enzyme Fundamentals” → “Enzyme Kinetics”).
- `graph_export.py`  
  - Export a graph for visualization (e.g., JSON for a frontend graph library: D3, Cytoscape, or similar).

### 2.3 Stage 3 – Query routing (adaptive retrieval)

- **Input**:  
  - User question (e.g., “Why does the enzyme’s active site shape matter for reaction rate, and how does that change with temperature?”)
- **Output**:  
  - A routing decision:  
    - `mode = "dense"` → use vector DB only.  
    - `mode = "graph"` → use graph‑based retrieval + multi‑hop.
    - `mode = "none"` → answer without retrieval.

Sub‑modules:

- `query_router.py`  
  - Uses a small LLM‑based classifier or heuristics:
    - If the question requires retrieving.
    - Single‑hop pattern: “What is X?”, “Define Y”, “Who invented Z?” → dense retrieval.
    - Multi‑hop pattern: “Why…”, “How does… change…”, “Compare A and B”, “Explain A in terms of B and C” → graph retrieval.
- `hop_classifier.py` (optional helper)  
  - Assigns a hop estimate to each question based on references to multiple concepts.

### 2.4 Stage 4 – Retrieval

Two sub‑paths:

#### 4.1 Dense retrieval (RAG)

- `vector_retriever.py`
  - Embed the query with same encoder used for chunks.
  - Query vector DB for top‑K chunks.
  - Return: `list[chunk_id, text, score]`.

#### 4.2 Graph‑based retrieval (GraphRAG)

- `graph_retriever.py`  
  - Step 1: extract key concepts from the query.
  - Step 2: find the starting concept nodes in the graph.
  - Step 3: perform **subgraph extraction** (e.g., 2–3 hops) around these nodes.
  - Step 4: from the subgraph, collect:
    - All nodes in the subgraph.
    - All edges between them.
    - All `source_chunk_id`s attached to these nodes/edges.
  - Step 5: fetch the raw text chunks from the document store.

***

## 3. Self‑RAG self‑reflection layer

### 3.1 Integration with retrieval

Implement an Agentic Evaluator (Corrective/Reflective RAG) to act as a verification gate. Force the LLM to output a structured JSON/Pydantic object before final generation:
  - needs_retrieval: boolean (decides when to retrieve)
  - is_relevant: boolean (is the retrieved context relevant to the question?)
  - is_supported: boolean (is the generated answer supported by the context?)
  - is_useful: boolean (does it directly answer the user's prompt?)

- `self_rag_reflector.py`  
  - Takes:
    - Question, retrieved context (from either dense or graph retrieval).
  - Generates:
    - A short reflection verdict (e.g., “ISSUP: YES”, “ISREL: NO”).
  - Rules:
    - If `ISREL = NO`, return to retrieval with a refined subgraph or rerun routing.
    - If `ISSUP = NO`, flag the answer as unreliable and optionally re‑prompt the model with stricter constraints.

This layer is your **verification gate** between retrieval and generation.

***

## 4. Multi‑hop reasoning chain

### 4.1 Reasoning pipeline

- `multi_hop_reasoner.py`  
  - For questions classified as 2+ hops:
    - **Decompose** the question into sub‑queries:
      - “Why does enzyme active site shape matter for reaction rate?”
      - “How does temperature affect reaction rate?”
      - “How are these two connected?”
    - For each sub‑query:
      - Run the same retrieval + reflection pipeline.
    - **Assemble** a **traceable evidence chain**:
      - Ordered list of supporting subgraphs and chunks.
  - Output:  
    - A structured reasoning trace (e.g., list of steps + supporting chunks / graph nodes).
    - This trace is what you will show in the UI.

***

## 5. Generation and answer composition

### 5.1 Prompt templates

- `prompts.py`  
  - `prompt_dense_rag` – for single‑hop:  
    ```
    Use the following context to answer the question.
    Context:
    {chunk_texts}
    Question:
    {question}
    ...
    ```
  - `prompt_graph_rag` – for multi‑hop, with reasoning trace:
    ```
    Use the following subgraph and reasoning steps to answer the question.
    {graph_triplet_descriptions}
    ...
    Question:
    {question}
    ... 
    ```
  - `prompt_self_rag` – includes reflection tokens:
    - Ask the model to prepend each answer with `ISREL`, `ISSUP`, `ISUSE`.

### 5.2 Answer generator

- `generator.py`
  - Calls your LLM (Claude, GPT‑4, Gemini, etc.) with:
    - Question.
    - Retrieved context (dense or graph‑based).
    - Appropriate prompt template.
  - Returns:
    - Answer text + reflection tokens.
    - Evidence trace (list of cited chunks and nodes).

***

## 6. UI / Visualization (graph mindmap)

The graph is rendered as an interactive concept mindmap with highlighted nodes, making the system's reasoning path transparent and observable.

### 6.1 Frontend modules

You can keep this separate from the core system:

- `graph_visualizer.py`
  - Given:
    - graph (nodes, edges) from the retrieval step.
  - Generates:
    - A visualization‑ready format (e.g., JSON for D3 / Cytoscape):
      - Each node has `id`, `name`, `highlighted: true/false`.
    - Highlighted nodes = nodes that actually contributed to the final answer.
- `frontend/`
  - A simple React + Typescript + Vite + Tailwind frontend that:
    - Shows the graph mindmap.
    - Lets the user click nodes to see the source text / chunk.

## 7. Evaluation and Research Design
### Research Hypothesis: 
Does augmenting a traditional dense RAG pipeline with a self-verifying, graph-based multi-hop retrieval system (LuminRAG) significantly improve factual grounding and answer accuracy on complex, university-level educational queries, while maintaining acceptable latency?

To prove this system is an advancement over the current state-of-the-art, the evaluation must rigorously benchmark LuminRAG against isolated baselines using both established academic datasets and a custom educational benchmark.

### 7.1 Baselines: 

| Baseline | System Architecture Description | Research Purpose |
|-----------|--------------------------------|------------------|
| **Naive RAG** | Standard Vector DB + dense embeddings + LLM generation. | Proves that standard retrieval fails on complex, multi-hop educational queries. |
| **Pure GraphRAG** | Graph-based multi-hop retrieval only (no Self-RAG verification gate). | Isolates the performance boost provided strictly by the knowledge graph structure. |
| **Pure Self-RAG** | Vector retrieval combined with the agentic reflection gate (no knowledge graph). | Isolates the impact of self-correction and evaluation from the graph traversal. |
| **LuminRAG (Full)** | The complete proposed pipeline: Multimodal Ingestion + Adaptive Routing + Graph/Vector Retrieval + Self-RAG Verification. | Establishes the performance of the state-of-the-art integrated system. |

### 7.2 Benchmark Datasets
HotpotQA: The industry gold standard for multi-hop question answering. It requires systems to parse and connect information across multiple Wikipedia documents to find the reasoning chain.

MuSiQue: A highly rigorous multi-hop dataset specifically designed to prevent language models from "cheating" via single-hop shortcuts or memorized knowledge.

ScienceQA: A multimodal dataset of science questions featuring text, images, and diagrams. This perfectly aligns with testing your multimodal chunking and ingestion pipeline.

QASper: A dataset of question-answering pairs over natural language processing research papers. Excellent for testing the system's ability to comprehend dense, academic language.

Lumin-Edu Benchmark (Custom): A manually curated dataset of 100–200 complex questions derived directly from your specific university lecture videos, slides, and textbook PDFs. This proves the system works on messy, real-world classroom data rather than just sanitized academic benchmarks.


### 7.3 Evaluation Metrics
  #### Retrieval Metrics:
    - Hit Rate @ K: Did the correct chunk or graph node appear in the top K retrieved contexts?
    - Mean Reciprocal Rank (MRR): How high up in the retrieval ranking was the most relevant factual evidence?
  #### Generation & Reflection Metrics:
    - Faithfulness (Hallucination Rate): Is the generated answer strictly supported by the retrieved context? (Maps directly to the system's internal is_supported flag).
    - Answer Relevance: Does the generated answer directly address the user's prompt without unnecessary tangents? (Maps to the internal is_useful flag).
    - Context Precision: Did the system retrieve only the necessary context, or did it pull in irrelevant noise?
***
