# MCP Ticket RAG

An AI-powered system for querying and summarizing GitHub issues and comments using a hybrid **MCP live retrieval + RAG architecture**, enabling structured issue lookup and semantic reasoning over ticket data.

---

## Technology Stack

- **Orchestration:** Python service layer with modular routing logic  
- **UI Layer:** Streamlit interactive interface  
- **LLM Runtime / Models:** Ollama  
  - `phi3-mini` – semantic resolver  
  - `mistral` – answer generation  
  - `llava-phi3` – vision / OCR processing  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector Store:** ChromaDB  
- **Data Access:** GitHub MCP server (live issue and comment retrieval)  
- **Language:** Python

---

## AI & System Design Patterns

- **Retrieval Augmented Generation (RAG)** for semantic issue search  
- **Hybrid MCP + RAG architecture** combining live API retrieval with vector search  
- **Semantic query routing** using a lightweight LLM resolver  
- **Modular AI service design** with independent components  
- **LLM orchestration pipeline** separating reasoning and generation models  
- **Streaming LLM execution with observability diagnostics**  
- **Tool-assisted retrieval via MCP server integration**

---

## Capabilities Demonstrated

- Building **LLM-driven developer productivity tools**
- Designing **hybrid retrieval architectures (live APIs + vector search)**
- Implementing **multi-model orchestration pipelines**
- Integrating **LLMs with external systems using MCP**
- Building **observable AI services with request diagnostics**
- Developing **modular AI engineering systems suitable for production extension**

---

## High-Level Flow

User → Streamlit UI → Semantic Resolver (phi3-mini) → Router →  
MCP Live Retrieval / Vector RAG → Answer Generation (Mistral) → Response

---

## Future Plans

- Introduce **multi-agent workflows** for issue investigation and root cause analysis  
- Add **automated issue triage and classification agents**  
- Improve **context compression and retrieval ranking for faster RAG responses**

---
