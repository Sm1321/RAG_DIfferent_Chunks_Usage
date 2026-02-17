# 📚 Chunking Strategies in RAG Systems

This document summarizes different chunking strategies, overlap configurations, and production considerations used in Retrieval-Augmented Generation (RAG) systems.

---


## 🔹 1. Chunking Strategies Comparison

| Strategy | How It Works | Advantages | Limitations | Recommended Use |
|-----------|-------------|------------|--------------|----------------|
| Character-Based | Splits text by fixed character length | Simple, fast baseline | Breaks semantic boundaries | Quick prototypes |
| Token-Based | Splits by token count | Aligns with LLM context limits | Ignores semantic meaning | Strict token control scenarios |
| Recursive Character | Splits by paragraph → sentence → word hierarchy | Preserves structure better | Still size-based | General-purpose RAG systems |
| Semantic Chunking | Splits based on embedding similarity (meaning shift) | High topic coherence | Higher cost, slower | Legal, research, structured documents |
| Section-Based | Splits using headers or document structure | Clean logical grouping | Requires structured input | Markdown, HTML, technical docs |
| Hybrid Chunking | Combines structural + semantic + size constraints | Balanced precision & control | More complex | Production-grade RAG |

---

## 🔹 2. Chunk Size Trade-Off

| Chunk Size | Behavior | Suitable For |
|-------------|----------|--------------|
| 200–300 | Fine-grained retrieval | QA-heavy systems |
| 400–600 | Balanced precision & context | Most RAG applications |
| 700–1000 | Broader context, fewer chunks | Narrative content |
| 1000+ | Lower precision, fewer embeddings | Summarization-focused tasks |

---

## 🔹 3. Chunk Overlap Strategy

| Overlap % | Effect | Recommendation |
|------------|--------|----------------|
| 0% | No shared context | Not recommended |
| 5–10% | Minimal continuity | Large chunk sizes |
| 10–20% | Balanced context preservation | Default for most systems |
| 30%+ | High redundancy | Special cases only |

**Recommended Rule:**  
`Overlap = 10–20% of chunk size`

Example:
- chunk_size = 500 → overlap = 50–100

---

## 🔹 4. Strategy by Document Type

| Document Type | Recommended Strategy |
|---------------|----------------------|
| Legal Contracts | Semantic + Size Cap |
| Technical Manuals | Recursive + Overlap |
| Research Papers | Semantic + Section |
| Blog Articles | Recursive |
| Code Documentation | Function/Section-Based |
| Noisy PDFs | Cleaning → Recursive |

---

## 🔹 5. Production Considerations

- Text cleaning significantly impacts semantic chunking quality.
- Larger chunk sizes reduce embedding cost but may reduce retrieval precision.
- Higher overlap increases context continuity but raises embedding redundancy.
- More chunks → higher embedding cost → higher latency.
- Hybrid chunking is recommended for production systems.

---

## 🔹 6. Advanced Concepts

- Lost-in-the-middle problem
- Parent-child chunking
- Metadata-aware chunking
- Hierarchical chunking
- Adaptive chunking
- Context window optimization


---

## 🔹 7. Industry Pattern (Recommended Setup)

Production RAG systems typically use:

- Section-based splitting  
- Semantic segmentation  
- Maximum chunk size constraint  
- 10–20% overlap  
- Metadata tagging  

---
