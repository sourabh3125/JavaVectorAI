# VectorDB — Java Edition 🟠

A complete port of the original C++ VectorDB project to **Java + Spring Boot**.
Implements HNSW, KD-Tree, and BruteForce search algorithms from scratch — with a full RAG pipeline powered by Ollama.

> Built as a learning project to understand how production vector databases work under the hood — in Java, your native language.

---

## Java vs C++ — What Changed

| Concept | C++ | Java |
|---|---|---|
| HTTP server | `cpp-httplib` (single-header) | Spring Boot + embedded Tomcat |
| JSON | Manual string building with `ostringstream` | Jackson (auto-serialization) |
| Data structures | `std::vector`, `std::unordered_map` | `ArrayList`, `HashMap` |
| Pairs | `std::pair<float, int>` | Java `record SearchResult(float, int)` |
| Lambdas | `[&](const Request& req) {...}` | `(param) -> expression` |
| Thread safety | `std::mutex` + `std::lock_guard` | `ReentrantLock` |
| Pointers | `KDNode* left`, `KDNode* right` | Object references `KDNode left` |
| Memory mgmt | `new` / `delete` manually | Garbage Collector handles it |
| Build | `g++ -std=c++17 main.cpp -o db` | `mvn package` |
| Run | `./db` | `java -jar target/vectordb-java-1.0.0.jar` |

---

## Project Structure

```
vectordb-java/
├── pom.xml                          ← Maven build file (like a Makefile)
├── src/main/
│   ├── java/com/vectordb/
│   │   ├── VectorDbApplication.java        ← main() — app entry point
│   │   ├── model/
│   │   │   ├── VectorItem.java             ← C++ struct VectorItem
│   │   │   └── DocItem.java                ← C++ struct DocItem
│   │   ├── algorithms/
│   │   │   ├── DistanceMetric.java         ← C++ DistFn typedef
│   │   │   ├── BruteForce.java             ← C++ class BruteForce
│   │   │   ├── KDTree.java                 ← C++ class KDTree
│   │   │   └── HNSW.java                   ← C++ class HNSW (the hard one)
│   │   ├── service/
│   │   │   ├── VectorDBService.java        ← C++ class VectorDB
│   │   │   ├── DocumentDBService.java      ← C++ class DocumentDB + RAG
│   │   │   └── OllamaClient.java           ← C++ class OllamaClient
│   │   ├── controller/
│   │   │   └── VectorController.java       ← C++ main() HTTP handlers
│   │   └── config/
│   │       └── WebConfig.java              ← CORS + static file serving
│   └── resources/
│       ├── application.properties          ← server port, Ollama settings
│       └── static/
│           └── index.html                  ← same frontend as original
```

---

## Prerequisites

1. **Java 17+** — download from https://adoptium.net
2. **Maven** — download from https://maven.apache.org (or use `./mvnw`)
3. **Ollama** — download from https://ollama.com

   ```bash
   ollama pull nomic-embed-text   # embedding model (~274 MB)
   ollama pull llama3.2           # LLM (~2 GB)
   ```

---

## Run the Project

```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. Build and run the Java app
cd vectordb-java
mvn spring-boot:run

# OR build a JAR first, then run it
mvn package
java -jar target/vectordb-java-1.0.0.jar

# 3. Open the browser
# http://localhost:8080
```

You should see:
```
=== VectorDB Java Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE  embed=nomic-embed-text  gen=llama3.2
```

---

## REST API (identical to C++ version)

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | Database stats |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all document chunks |
| `DELETE` | `/doc/delete/:id` | — | Delete a chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status |

---

## Key Java Concepts Used

### Spring Boot Annotations
- `@SpringBootApplication` — entry point + component scan
- `@Service` — marks a class as a Spring-managed singleton
- `@RestController` — marks a class as a REST API controller
- `@GetMapping`, `@PostMapping`, `@DeleteMapping` — HTTP method routing
- `@RequestParam` — query parameter extraction
- `@PathVariable` — URL path variable extraction
- `@RequestBody` — JSON body deserialization
- `@CrossOrigin` — CORS permission
- `@PostConstruct` — run after Spring wires dependencies (for loadDemo)
- `@Value` — inject values from application.properties

### Java Language Features
- **Records** — concise immutable data classes (replaces C++ structs used as return types)
- **Switch expressions** — `return switch (algo) { case "hnsw" -> ...; }`
- **Text blocks** — multi-line strings
- **var** — local type inference
- **Generics** — `List<Float>`, `Map<Integer, Node>` (safer than C++ templates)
- **Functional interface** — `@FunctionalInterface DistanceMetric` (like C++ `std::function`)
- **Lambda** — `(a, b) -> { ... }` (like C++ lambdas without capture syntax)

---

## How to Push This to GitHub as Your Own

```bash
cd vectordb-java

# Initialize a new git repository
git init
git add .
git commit -m "Initial commit: VectorDB Java - HNSW, KD-Tree, BruteForce + RAG"

# Create a new repo on GitHub (github.com → New repository)
# Do NOT initialize with README, .gitignore, or license

# Connect and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

---

## License

MIT — use however you want.
