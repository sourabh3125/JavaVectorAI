# JavaVectorAI 

> A production-grade **Vector Database** built from scratch in Java with HNSW, KD-Tree & BruteForce algorithms + full **RAG Pipeline** using local AI via Ollama.

## What This Project Does

Normal databases search by exact keywords.  
This searches by **meaning** — just like ChatGPT.

Built the same technology used by **Pinecone, Weaviate, Chroma** — from scratch in Java.

## Features

- 3 Search Algorithms from scratch — BruteForce, KD-Tree, HNSW
- 3 Distance Metrics — Euclidean, Cosine, Manhattan
- RAG Pipeline — Insert docs → Ask questions → Get AI answers
- Local AI via Ollama — no API key, no cost, runs offline
- REST API with Spring Boot
- 2D Vector Visualization (PCA projection)
- Real-time Algorithm Benchmarking

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Java 17 | Core language |
| Framework | Spring Boot 3.2 | REST API + Embedded Tomcat |
| Build Tool | Maven | Dependency management |
| AI Runtime | Ollama 0.24 | Local LLM inference |
| Embed Model | nomic-embed-text | Text → 768D vectors |
| LLM | llama3.2 (2B) | Answer generation |
| Search Algo | HNSW (custom) | Production-grade vector search |
| HTTP Client | java.net.http | Ollama API calls |
| Concurrency | ReentrantLock | Thread-safe operations |

#How It Works

#Vector Embeddings
