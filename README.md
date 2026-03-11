# Mini-RAG-for-Knowledge-Base-Answering
### Semantic Search + Generative AI for Customer Support Q&A

A lightweight Retrieval-Augmented Generation (RAG) system built in Google Colab that answers customer support questions by intelligently retrieving relevant knowledge base articles and generating accurate, contextual responses using the Gemini API.

For original project code, pls check out: https://colab.research.google.com/drive/1Zs4bnM1lTYcQImoVaLKMSrPx7R7UA0H5?usp=sharing#scrollTo=mv3qpWzYwsDW
---

##  Overview

This project implements a full RAG pipeline tailored for a customer support knowledge base (Hiver). Given a natural language query, the system:

1. **Retrieves** the most semantically relevant KB articles using sentence embeddings
2. **Ranks** them by cosine similarity and calculates a confidence score
3. **Generates** a concise, grounded answer using Gemini 2.0 Flash
4. **Handles** out-of-scope queries gracefully with low-confidence flagging

---

##  Architecture
```
User Query
    │
    ▼
┌─────────────────────────┐
│  Sentence Transformer   │  ← all-MiniLM-L6-v2
│  (Query Embedding)      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Cosine Similarity      │  ← Against pre-indexed KB embeddings
│  Search (Top-K)         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Confidence Scoring     │  ← Based on similarity + score gap
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Gemini 2.0 Flash       │  ← Context-grounded answer generation
│  (Answer Generation)    │
└──────────┬──────────────┘
           │
           ▼
    Structured Response
  (Answer + Confidence + Sources)
```

---

##  Features

- **Semantic retrieval** using `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Gemini-powered generation** for natural, accurate answers
- **Confidence scoring** with a visual progress bar output
- **Graceful failure handling** for out-of-scope queries (similarity threshold: `< 0.3`)
- **Fallback extractive QA** if the Gemini API is unavailable
- **10 pre-loaded KB articles** covering Hiver features (Automations, CSAT, SLAs, Analytics, and more)
- **Exportable results** as structured JSON

---

### Prerequisites

- Python 3.8+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
- Google Colab (recommended) or a local Jupyter environment

### Installation
```bash
pip install sentence-transformers numpy google-generativeai
```

### Configuration

Open `RAG_Flow.py` (or the Colab notebook) and replace the placeholder with your API key:
```python
GEMINI_API_KEY = "your-api-key-here"
```

>  **Security Note:** Do not commit your API key to version control. Use environment variables or Colab Secrets in production.

### Running

Execute the cells in order, or run the script directly:
```bash
python RAG_Flow.py
```

---

##  Test Queries

The system is validated with three queries out of the box:

| # | Query | Expected Outcome |
|---|-------|-----------------|
| 1 | `"How do I configure automations in Hiver?"` |  High confidence answer |
| 2 | `"Why is CSAT not appearing?"` |  High confidence answer |
| 3 | `"How do I integrate with Salesforce?"` |  Low confidence — out-of-scope |

---

##  Knowledge Base

The system ships with **10 KB articles** covering:

- Getting Started with Hiver Automations
- Understanding & Troubleshooting CSAT
- Setting Up SLAs
- Advanced Automation Rules
- Analytics and Reporting
- Email Tagging and Organization
- User Management and Permissions
- Mail Merge Feature Guide
- Workflow Rules and Triggers

---

##  Core Class: `GeminiRAG`
```python
rag = GeminiRAG(model_name='all-MiniLM-L6-v2', gemini_model='gemini-2.0-flash')
rag.index_articles(KB_ARTICLES)

result = rag.query("How do I configure automations?", top_k=3)
# Returns: { query, retrieved_articles, answer, confidence, reasoning }
```

### Key Methods

| Method | Description |
|--------|-------------|
| `index_articles(articles)` | Embeds and indexes all KB articles |
| `retrieve(query, top_k)` | Returns top-K articles by cosine similarity |
| `calculate_confidence(score, articles)` | Computes a confidence score (0–1) |
| `generate_answer_with_gemini(query, articles)` | Generates a grounded answer via Gemini |
| `query(question, top_k)` | Full RAG pipeline in one call |

---

##  Retrieval Quality Improvements

Five production-ready enhancements are documented in the codebase:

1. **Hybrid Search** — Combine dense embeddings with BM25 keyword matching (+15–20% precision)
2. **Chunk-Level Retrieval + Reranking** — Two-stage bi-encoder / cross-encoder pipeline (+25–30% precision)
3. **Query Expansion (HyDE)** — LLM-generated alternative phrasings and hypothetical answers (+20–25% recall)
4. **Metadata Filtering** — Filter articles by plan tier, user role, and feature flags (−30–40% irrelevant results)
5. **Feedback Loop** — Collect click/rating signals to fine-tune embeddings monthly (+15–20% accuracy over time)

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarity | NumPy cosine similarity |
| Generation | Google Gemini 2.0 Flash |
| Environment | Google Colab / Python 3.8+ |

---
