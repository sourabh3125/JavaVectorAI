package com.vectordb.controller;

import com.vectordb.algorithms.HNSW;
import com.vectordb.model.DocItem;
import com.vectordb.model.VectorItem;
import com.vectordb.service.DocumentDBService;
import com.vectordb.service.OllamaClient;
import com.vectordb.service.VectorDBService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * REST API Controllers — maps HTTP endpoints to service methods.
 *
 * In C++, all HTTP routing was done in main() using cpp-httplib:
 *   svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) { ... });
 *
 * In Java Spring Boot, we use annotations:
 *   @RestController + @GetMapping/@PostMapping/@DeleteMapping
 *
 * Spring Boot automatically handles:
 *   - JSON serialization of return values (Jackson)
 *   - CORS configuration (via @CrossOrigin)
 *   - Request body parsing (@RequestBody)
 *   - Path variables (@PathVariable)
 *   - Query parameters (@RequestParam)
 *
 * All the manual JSON building (ostringstream ss; ss << "{...}") from C++ is GONE.
 * Just return a Java object (Map, record, POJO) and Jackson converts it to JSON.
 */
@RestController
@CrossOrigin(origins = "*")  // C++ equivalent: cors(res) in every handler
public class VectorController {

    private final VectorDBService   vectorDB;
    private final DocumentDBService documentDB;
    private final OllamaClient      ollama;

    // Constructor injection — Spring wires these automatically
    public VectorController(VectorDBService vectorDB, DocumentDBService documentDB,
                            OllamaClient ollama) {
        this.vectorDB   = vectorDB;
        this.documentDB = documentDB;
        this.ollama     = ollama;
    }

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────
    // Equivalent to C++: svr.Get("/search", ...) etc.

    /**
     * GET /search?v=0.1,0.2,...&k=5&metric=cosine&algo=hnsw
     *
     * C++ equivalent:
     *   svr.Get("/search", [&](const Request& req, Response& res) {
     *       auto q = parseVec(req.get_param_value("v"));
     *       ...
     *   });
     *
     * @RequestParam maps "?v=..." query params to Java variables automatically.
     */
    @GetMapping("/search")
    public ResponseEntity<?> search(
        @RequestParam("v") String vectorStr,
        @RequestParam(defaultValue = "5") int k,
        @RequestParam(defaultValue = "cosine") String metric,
        @RequestParam(defaultValue = "hnsw") String algo
    ) {
        List<Float> query = parseVector(vectorStr);

        if (query.size() != vectorDB.getDims()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "need " + vectorDB.getDims() + "D vector"));
        }

        VectorDBService.SearchResult result = vectorDB.search(query, k, metric, algo);

        // Build JSON response — C++ used manual string building with ostringstream
        // Java: just return a Map and Spring/Jackson serializes it automatically
        List<Map<String, Object>> resultList = new ArrayList<>();
        for (VectorDBService.Hit hit : result.hits()) {
            resultList.add(Map.of(
                "id",        hit.id(),
                "metadata",  hit.metadata(),
                "category",  hit.category(),
                "distance",  hit.distance(),
                "embedding", hit.embedding()
            ));
        }

        return ResponseEntity.ok(Map.of(
            "results",   resultList,
            "latencyUs", result.latencyUs(),
            "algo",      result.algo(),
            "metric",    result.metric()
        ));
    }

    /**
     * POST /insert
     * Body: {"metadata": "...", "category": "...", "embedding": [0.1, 0.2, ...]}
     *
     * @RequestBody automatically deserializes the JSON body into a Map.
     * C++: parseBody(req.body, meta, cat, emb) — manual JSON parsing
     */
    @PostMapping("/insert")
    public ResponseEntity<?> insert(@RequestBody Map<String, Object> body) {
        String metadata  = (String) body.get("metadata");
        String category  = (String) body.getOrDefault("category", "");
        List<?> rawEmb   = (List<?>) body.get("embedding");

        if (metadata == null || rawEmb == null) {
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body"));
        }

        List<Float> embedding = rawEmb.stream()
            .map(v -> ((Number) v).floatValue())
            .toList();

        if (embedding.size() != vectorDB.getDims()) {
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body"));
        }

        int id = vectorDB.insert(metadata, category, embedding);
        return ResponseEntity.ok(Map.of("id", id));
    }

    /**
     * DELETE /delete/{id}
     *
     * C++ equivalent:
     *   svr.Delete(R"(/delete/(\d+))", [&](const Request& req, Response& res) {
     *       int id = stoi(req.matches[1]);
     *       ...
     *   });
     *
     * @PathVariable extracts the {id} from the URL path.
     */
    @DeleteMapping("/delete/{id}")
    public ResponseEntity<?> delete(@PathVariable int id) {
        boolean ok = vectorDB.delete(id);
        return ResponseEntity.ok(Map.of("ok", ok));
    }

    /** GET /items — list all demo vectors */
    @GetMapping("/items")
    public ResponseEntity<?> items() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (VectorItem item : vectorDB.getAll()) {
            result.add(Map.of(
                "id",        item.getId(),
                "metadata",  item.getMetadata(),
                "category",  item.getCategory(),
                "embedding", item.getEmbedding()
            ));
        }
        return ResponseEntity.ok(result);
    }

    /** GET /benchmark?v=...&k=5&metric=cosine — compare all 3 algorithms */
    @GetMapping("/benchmark")
    public ResponseEntity<?> benchmark(
        @RequestParam("v") String vectorStr,
        @RequestParam(defaultValue = "5") int k,
        @RequestParam(defaultValue = "cosine") String metric
    ) {
        List<Float> query = parseVector(vectorStr);
        if (query.size() != vectorDB.getDims()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "need " + vectorDB.getDims() + "D vector"));
        }

        VectorDBService.BenchmarkResult bench = vectorDB.benchmark(query, k, metric);
        return ResponseEntity.ok(Map.of(
            "bruteforceUs", bench.bruteforceUs(),
            "kdtreeUs",     bench.kdtreeUs(),
            "hnswUs",       bench.hnswUs(),
            "itemCount",    bench.itemCount()
        ));
    }

    /** GET /hnsw-info — graph structure details */
    @GetMapping("/hnsw-info")
    public ResponseEntity<?> hnswInfo() {
        HNSW.GraphInfo info = vectorDB.getHnswInfo();
        return ResponseEntity.ok(Map.of(
            "topLayer",      info.topLayer,
            "nodeCount",     info.nodeCount,
            "nodesPerLayer", info.nodesPerLayer,
            "edgesPerLayer", info.edgesPerLayer,
            "nodes",         info.nodes,
            "edges",         info.edges
        ));
    }

    /** GET /stats */
    @GetMapping("/stats")
    public ResponseEntity<?> stats() {
        return ResponseEntity.ok(Map.of(
            "count",      vectorDB.size(),
            "dims",       vectorDB.getDims(),
            "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
            "metrics",    List.of("euclidean", "cosine", "manhattan")
        ));
    }

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    /**
     * POST /doc/insert
     * Body: {"title": "...", "text": "..."}
     *
     * Chunks the text, embeds each chunk via Ollama, stores in DocumentDB.
     */
    @PostMapping("/doc/insert")
    public ResponseEntity<?> docInsert(@RequestBody Map<String, String> body) {
        String title = body.get("title");
        String text  = body.get("text");

        if (title == null || title.isBlank() || text == null || text.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "need title and text"));
        }

        try {
            List<Integer> ids = documentDB.insertDocument(title, text);
            return ResponseEntity.ok(Map.of(
                "ids",    ids,
                "chunks", ids.size(),
                "dims",   documentDB.getDims()
            ));
        } catch (IllegalStateException e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    /** DELETE /doc/delete/{id} */
    @DeleteMapping("/doc/delete/{id}")
    public ResponseEntity<?> docDelete(@PathVariable int id) {
        boolean ok = documentDB.delete(id);
        return ResponseEntity.ok(Map.of("ok", ok));
    }

    /** GET /doc/list */
    @GetMapping("/doc/list")
    public ResponseEntity<?> docList() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (DocItem doc : documentDB.getAll()) {
            String preview = doc.getText().length() > 120
                ? doc.getText().substring(0, 120) + "…"
                : doc.getText();
            long wordCount = Arrays.stream(doc.getText().split("\\s+")).count();
            result.add(Map.of(
                "id",      doc.getId(),
                "title",   doc.getTitle(),
                "preview", preview,
                "words",   wordCount
            ));
        }
        return ResponseEntity.ok(result);
    }

    /**
     * POST /doc/ask
     * Body: {"question": "...", "k": 3}
     *
     * Full RAG pipeline: embed question → retrieve → generate answer.
     */
    @PostMapping("/doc/ask")
    public ResponseEntity<?> docAsk(@RequestBody Map<String, Object> body) {
        String question = (String) body.get("question");
        int k = body.containsKey("k") ? ((Number) body.get("k")).intValue() : 3;

        if (question == null || question.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "need question"));
        }

        try {
            DocumentDBService.RagResult result = documentDB.ask(question, k);

            List<Map<String, Object>> contexts = new ArrayList<>();
            for (DocumentDBService.DocSearchResult ctx : result.contexts()) {
                contexts.add(Map.of(
                    "id",       ctx.doc().getId(),
                    "title",    ctx.doc().getTitle(),
                    "text",     ctx.doc().getText(),
                    "distance", ctx.distance()
                ));
            }

            return ResponseEntity.ok(Map.of(
                "answer",   result.answer(),
                "model",    result.model(),
                "contexts", contexts,
                "docCount", result.docCount()
            ));
        } catch (IllegalStateException e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    /** GET /status — Ollama status and model info */
    @GetMapping("/status")
    public ResponseEntity<?> status() {
        boolean up = ollama.isAvailable();
        return ResponseEntity.ok(Map.of(
            "ollamaAvailable", up,
            "embedModel",      ollama.getEmbedModel(),
            "genModel",        ollama.getGenModel(),
            "docCount",        documentDB.size(),
            "docDims",         documentDB.getDims(),
            "demoDims",        vectorDB.getDims(),
            "demoCount",       vectorDB.size()
        ));
    }

    // ── HELPER ────────────────────────────────────────────────────────

    /**
     * Parse a comma-separated string of floats into a List<Float>.
     *
     * C++ equivalent:
     *   vector<float> parseVec(const string& s) {
     *       vector<float> v;
     *       istringstream ss(s); string t;
     *       while (getline(ss, t, ',')) v.push_back(stof(t));
     *       return v;
     *   }
     */
    private List<Float> parseVector(String input) {
        List<Float> result = new ArrayList<>();
        for (String part : input.split(",")) {
            try {
                result.add(Float.parseFloat(part.trim()));
            } catch (NumberFormatException ignored) {}
        }
        return result;
    }
}
