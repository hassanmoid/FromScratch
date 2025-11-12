# 📚 Local Multi-Index RAG (Retrieval-Augmented Generation)

This folder contains two complementary local RAG pipelines that let you query heterogeneous document collections (CVs, financial records, reimbursement policies, product specs) using a locally hosted quantized LLM (Mistral 7B GGUF) plus SentenceTransformer embeddings and FAISS vector stores.

You can:
- Maintain separate indexes per category (multi-index) to improve relevance.
- Run everything fully offline (after initial model + embedding downloads).
- Swap embedding models and tune chunking for precision vs. recall.
- Use either LlamaIndex (`rag_local_v2.ipynb`) or LangChain (`rag_local_langchain.ipynb`) depending on preferred abstractions.

---

## 📁 Directory Layout

```
local_RAG/
  data/
    CV/                <- Resumes & profiles (.pdf/.docx/.txt)
    financial/         <- Finance docs, statements
    reimbursement/     <- Policy / process docs
    specs/             <- Product spec sheets
  scripts/
    rag_local.ipynb            <- (legacy / initial draft)
    rag_local_v2.ipynb         <- LlamaIndex + custom FAISS (multi-index)
    rag_local_langchain.ipynb  <- LangChain pipeline (multi-index)
    chroma_db/                 <- (if used for Chroma experiments)
    embeddings/                <- Cached embeddings (LangChain notebook)
    faiss_indexes/             <- Saved FAISS stores (LangChain)
    models/                    <- Quantized GGUF LLM files (e.g. mistral-7b-instruct*.gguf)
```

---

## 🧪 Two Pipelines at a Glance

| Feature | `rag_local_v2.ipynb` (LlamaIndex) | `rag_local_langchain.ipynb` (LangChain) |
|---------|------------------------------------|------------------------------------------|
| Chunking | `SentenceSplitter` (semantic sentences) | `RecursiveCharacterTextSplitter` |
| Chunk size (example) | 1024 chars (overlap 192) | 16384 chars (overlap 4096; very large) |
| Embeddings | `BAAI/bge-small-en-v1.5` (GPU) | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | Manual FAISS (per category) | LangChain FAISS wrapper (per category) |
| LLM Integration | `LlamaCPP` (llama-index-llms-llama-cpp) | `LlamaCpp` (LangChain community) |
| Retrieval | Custom FAISS search (inner product) | Retriever via `.as_retriever(k=...)` |
| Multi-index | Yes (loop over category groups) | Yes (build FAISS for each category) |
| Source return | Manual design (extend easily) | Built-in `return_source_documents=True` |

---

## 🔧 Environment Setup (Windows, PowerShell)

### 1. Create and activate a virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Core Python dependencies
(Install only what you need; both stacks shown.)

```powershell
pip install sentence-transformers faiss-cpu chromadb pypdf docx2txt python-pptx
pip install llama-index llama-index-llms-llama-cpp
pip install langchain langchain-community langchain-text-splitters langchain-huggingface
pip install transformers accelerate
```

### 3. (Optional) GPU-enabled `llama-cpp-python`
If you want CUDA acceleration for Mistral 7B inference:

```powershell
# Prerequisites (outside pip):
# - Visual Studio 2022 with "Desktop development with C++"
# - CUDA Toolkit installed (match your driver & GPU)
# - CMake installed and added to PATH

setx FORCE_CMAKE 1
setx CMAKE_ARGS "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
# Re-open shell AFTER setx (or use $env: vars for current session):
$env:FORCE_CMAKE="1"
$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"

pip install --force-reinstall --no-cache-dir --verbose ^
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 llama-cpp-python
```

(Adjust `-DCMAKE_CUDA_ARCHITECTURES` to match your GPU’s SM version; 89 = Ada Lovelace. For Ampere use 86, etc.)

### 4. Verify installation
```powershell
python -c "from llama_cpp import Llama; print('Loaded OK')"
```

---

## 🤖 Model Download (Mistral 7B Instruct Quantized)

Choose a quantized GGUF file (e.g., `mistral-7b-instruct-v0.2.Q4_K_M.gguf`) and place it under `local_RAG/scripts/models/` or adjust `MODEL_PATH` variables.

Sources:
- https://huggingface.co/TheBloke
- https://huggingface.co/mistralai

Quantization guidance:
- Q4_K_M: Good balance of quality and memory (~4.5–5 GB RAM)
- Q5_K_M: Slightly better quality, more RAM
- Q8: Highest quality but heavier

---

## 🗂 Data Preparation

Place documents in the respective category folders under `local_RAG/data/`:
- Supported formats (as configured): `.pdf`, `.docx`, `.txt`
- Subfolders are scanned recursively.
- Add a new category by making a folder and updating the mapping in the notebook:
  - LlamaIndex: `DATA_PATHS_MAP`
  - LangChain: `CATEGORY_PATHS`

---

## ▶️ Running the LlamaIndex Pipeline (`rag_local_v2.ipynb`)

### Sequence of operations
1. Configuration constants (paths, chunking params, model parameters).
2. Load documents per category via `SimpleDirectoryReader(recursive=True)`.
3. Chunk using `SentenceSplitter(chunk_size=1024, chunk_overlap=192)` — sentence-aware splitting.
4. Embed all chunks in batches using `SentenceTransformer("BAAI/bge-small-en-v1.5")` (CUDA if available).
5. Build per-category FAISS indexes (normalized vectors → inner product ≈ cosine).
6. (Extend) Add retrieval + LLM answer synthesis with `LlamaCPP` (Mistral GGUF).
7. Query across targeted categories or aggregate if needed.

### Key tunables
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Trade-off between context completeness and vector count.
- `EMBED_MODEL_NAME_2`: Swap to `"all-MiniLM-L6-v2"` if GPU memory is limited.
- `TOP_K_RESULTS`: Number of retrieved chunks for answer context.
- `LLM_N_GPU_LAYERS`: Adjust downward if VRAM is limited (e.g., 20–28).
- `LLM_N_CTX`: Real Mistral Instruct context ≈ 8K; values beyond may not take effect.

### Example modification: Add a new category
```python
DATA_PATHS_MAP["LEGAL"] = "../data/legal"
grouped_documents = load_documents(DATA_PATHS_MAP)
```

---

## ▶️ Running the LangChain Pipeline (`rag_local_langchain.ipynb`)

### Sequence of operations
1. Load & verify device (CUDA vs CPU).
2. Build `CATEGORY_PATHS` mapping.
3. Load and chunk documents with `RecursiveCharacterTextSplitter` (current config uses very large chunk sizes: 16384 / overlap 4096).
4. Initialize `HuggingFaceEmbeddings` (MiniLM).
5. Build and persist FAISS indexes per category (`faiss_indexes/<category>/`).
6. Load quantized Mistral via `LlamaCpp`.
7. Create RAG chains (`RetrievalQA`) with custom prompt template to reduce hallucination.
8. Run sample queries; optionally return source docs.

### Recommended Adjustments
Large chunk size (16384) can:
- Reduce number of vectors (faster) but risk lower granularity.
- Increase context noise per retrieved chunk.
Consider downsizing to something like:
```python
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
```

### Query example
```python
cv_chain = rag_chains["CV"]
result = cv_chain.invoke({"query": "Where has Moid Hassan worked in the past?"})
print(result["result"])
for doc in result["source_documents"]:
    print(doc.page_content[:300])
```

---

## 💬 Prompt Design

LangChain RAG prompt (strict version):
```
You are a precise and intelligent assistant that answers questions. Use ONLY the following retrieved context...
If you are not certain or answer is not in the provided Context, say "I don't know based on the provided context."
```

Improvement ideas:
- Add explicit instruction to list sources.
- Include JSON output format for structured extraction tasks.
- Penalize unsupported speculation explicitly.

---

## 🎯 Retrieval & Similarity

- FAISS index uses inner product (`IndexFlatIP`) after L2 normalization → cosine similarity.
- For BGE / MiniLM embeddings this is appropriate.
- If switching to embedding models with unnormalized use-cases, ensure you preserve semantics (e.g., use `IndexFlatL2` with non-normalized embeddings).

---

## 🧪 Sample Queries (Provided in Notebooks)

CV:
- “What programming languages does Moid Hassan know?”
- “Where is Moid Hassan currently working?”

Specs:
- “Which GPU Surface Laptop Studio 2 has?”
- “Suggest a laptop which comes with a dedicated GPU…”

Financial:
- “What is the total dividend income in August 2025?”

---

## ⚙️ Customization Cheatsheet

| Goal | Change |
|------|--------|
| Faster embedding | Use `all-MiniLM-L6-v2` or `bge-small` |
| Higher quality retrieval | Decrease chunk size; increase overlap modestly |
| Lower memory footprint | Reduce `LLM_N_GPU_LAYERS`; use Q4 quantization |
| More context per answer | Increase `TOP_K_RESULTS` or merge categories |
| Minimize hallucinations | Stricter prompt + return sources + temperature ≤ 0.2 |
| Add new file type | Extend loader maps (LangChain notebook `EXT_TO_LOADER`) |

---

## 🛠 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `CUDA error: invalid device ordinal` | Wrong architecture flag | Adjust `CMAKE_ARGS` arch to match GPU SM |
| `DLL load failed` for `llama_cpp` | Missing MSVC / Build Tools | Install VS 2022 C++ workload & restart shell |
| OOM during embedding | Batch too large | Lower `batch_size` in `chunk_and_embed` |
| Slow queries | Too many large chunks | Reduce `CHUNK_SIZE` / clean irrelevant docs |
| Empty retrieval results | Index not built or wrong path | Re-run index build cells; verify folder paths |
| LLM ignores big `N_CTX` | Model capped at 8K | Keep realistic context (≤ 8192) |

---

## 🚀 Performance Tips

- Normalize embeddings before FAISS IP search (done in LlamaIndex pipeline).
- Batch embedding (`embed_model.encode(list_of_texts, batch_size=128)`).
- Persist FAISS indexes to skip rebuilds (LangChain already does).
- Use smaller embedding models during prototyping; upgrade later (e.g., `bge-base-en`).
- For multiple concurrent queries, wrap retrieval + generation in async tasks (future enhancement).

---

## 🔍 Extending the System

1. Add Category:
   - Create folder under `data/<NewCategory>`
   - Update mapping in notebook
2. Add Multi-Category Query:
   - Retrieve from each FAISS index separately
   - Merge top-k sets (rank by similarity score)
3. Hybrid Retrieval:
   - Add BM25 (e.g., `rank_bm25`) + vector fusion (weighted scoring)
4. Metadata Filtering:
   - Store richer metadata (doc type, date) and filter before similarity search.

---

## ✅ Validation Checklist Before Querying

- [ ] Model file present at `models/<mistral-*.gguf>`
- [ ] Embedding model downloaded (first run)
- [ ] Documents loaded: counts logged per category
- [ ] Chunks created (non-zero for target categories)
- [ ] FAISS indexes built (`ntotal > 0`)
- [ ] Test prompt returns coherent answer
- [ ] Retrieval returns relevant source snippets

---

## 🔐 Privacy & Locality

All embedding + inference operations happen locally:
- No external API calls after initial HuggingFace model downloads.
- Safe for confidential document experimentation (be mindful of transient notebook output).

---

## 🧭 Roadmap Ideas

- CLI wrapper for batch Q&A
- Simple Streamlit UI for interactive browsing
- Automatic evaluation: similarity vs answer correctness
- Structured extraction mode (JSON schema validation)
- Caching retrieved contexts for repeated queries

---

## ❓ FAQ

**Q: Why are my answers generic?**  
Chunk too large → irrelevant filler in retrieved context. Decrease chunk size.

**Q: Can I use a different LLM (e.g., Phi-3, Llama 3)?**  
Yes—replace model file + adjust params. Ensure quantized GGUF format if using `llama-cpp`.

**Q: How do I get source file names in LlamaIndex pipeline?**  
Persist `doc.metadata` when building nodes; add a mapping `id -> source` when constructing the FAISS store.

**Q: Should I use Chroma instead of FAISS?**  
FAISS is simpler and fast for pure vector similarity. Chroma adds persistence & metadata querying—use if you need filtering or embeddings management.

---

## 🧩 Minimal Example (LangChain Quick Start)

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(["Alpha", "Beta about GPUs", "Gamma about finance"], embedding=embedding_model)
llm = LlamaCpp(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_gpu_layers=-1, max_tokens=256)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 2}))
print(qa.invoke({"query": "Which text mentions GPUs?"})["result"])
```

---

## 🏁 Summary

This `local_RAG` module equips you with two robust, local, multi-index retrieval pipelines:
- LlamaIndex version for granular control over chunking & raw FAISS operations.
- LangChain version for rapid composition of retrieval + generation chains with source tracing.

Tune chunk sizes, embedding models, and retrieval depth to optimize relevance. Extend categories effortlessly and keep everything offline for privacy.

Happy experimenting! 🔍🤖
