package com.vectordb.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * OllamaClient — calls Ollama's local REST API for embeddings and text generation.
 *
 * C++ equivalent: class OllamaClient { ... httplib::Client ... }
 *
 * Key Java vs C++ differences:
 *   C++ uses cpp-httplib (single-header HTTP library)
 *   Java uses java.net.http.HttpClient (built-in since Java 11)
 *
 *   C++ manually parses JSON with string operations
 *   Java uses Jackson (industry-standard JSON library)
 *
 *   C++ uses raw string building for request bodies
 *   Java uses Jackson ObjectMapper to build JSON safely
 *
 * Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
 */
@Service
public class OllamaClient {

    private static final Logger log = LoggerFactory.getLogger(OllamaClient.class);

    @Value("${ollama.host:http://127.0.0.1:11434}")
    private String host;

    @Value("${ollama.embed-model:nomic-embed-text}")
    private String embedModel;

    @Value("${ollama.gen-model:llama3.2}")
    private String genModel;

    // Java 11+ built-in HTTP client — equivalent to C++ httplib::Client
    private final HttpClient httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(3))
        .build();

    // Jackson — handles JSON serialization/deserialization automatically
    // In C++ this was done manually with string parsing
    private final ObjectMapper mapper = new ObjectMapper();

    // ── AVAILABILITY CHECK ────────────────────────────────────────────

    /**
     * Check if Ollama is running.
     *
     * C++ equivalent:
     *   bool isAvailable() {
     *       httplib::Client cli(host, port);
     *       cli.set_connection_timeout(2, 0);
     *       auto res = cli.Get("/api/tags");
     *       return res && res->status == 200;
     *   }
     */
    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(host + "/api/tags"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ── EMBED ─────────────────────────────────────────────────────────

    /**
     * Generate a vector embedding for the given text using Ollama.
     *
     * API call: POST /api/embeddings
     * Body: {"model": "nomic-embed-text", "prompt": "your text here"}
     * Response: {"embedding": [0.123, 0.456, ...]}  (768 floats)
     *
     * C++ equivalent:
     *   vector<float> embed(const string& text) {
     *       httplib::Client cli(host, port);
     *       string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
     *       auto res = cli.Post("/api/embeddings", body, "application/json");
     *       if (!res || res->status != 200) return {};
     *       return parseEmbedding(res->body);
     *   }
     *
     * Java differences:
     *   - Jackson builds the JSON body safely (no manual string escaping!)
     *   - Jackson parses the response automatically
     *   - Checked exceptions require try/catch
     */
    public List<Float> embed(String text) {
        try {
            // Build request body using Jackson — no manual string escaping needed
            String requestBody = mapper.writeValueAsString(
                mapper.createObjectNode()
                    .put("model", embedModel)
                    .put("prompt", text)
            );

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(host + "/api/embeddings"))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                log.warn("Ollama embed returned status {}", response.statusCode());
                return List.of();
            }

            // Parse response using Jackson — C++ did this manually with string operations
            JsonNode root = mapper.readTree(response.body());
            JsonNode embeddingArray = root.get("embedding");

            if (embeddingArray == null || !embeddingArray.isArray()) return List.of();

            List<Float> embedding = new ArrayList<>(embeddingArray.size());
            for (JsonNode node : embeddingArray) {
                embedding.add((float) node.asDouble());
            }
            return embedding;

        } catch (IOException | InterruptedException e) {
            log.error("Ollama embed failed: {}", e.getMessage());
            return List.of();
        }
    }

    // ── GENERATE ─────────────────────────────────────────────────────

    /**
     * Generate a text response from Ollama using the LLM.
     *
     * API call: POST /api/generate
     * Body: {"model": "llama3.2", "prompt": "...", "stream": false}
     * Response: {"response": "the generated text", ...}
     *
     * stream: false means we wait for the complete response before returning.
     * (The C++ original also used stream:false)
     *
     * C++ equivalent:
     *   string generate(const string& prompt) {
     *       cli.set_read_timeout(180, 0);  // LLMs can be slow
     *       string body = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + esc(prompt) + "\",\"stream\":false}";
     *       auto res = cli.Post("/api/generate", body, "application/json");
     *       ...
     *   }
     */
    public String generate(String prompt) {
        try {
            String requestBody = mapper.writeValueAsString(
                mapper.createObjectNode()
                    .put("model", genModel)
                    .put("prompt", prompt)
                    .put("stream", false)
            );

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(host + "/api/generate"))
                .timeout(Duration.ofSeconds(180))  // LLMs can take a while on CPU
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                return "ERROR: Ollama returned status " + response.statusCode() +
                       ". Make sure Ollama is running: ollama serve";
            }

            JsonNode root = mapper.readTree(response.body());
            JsonNode responseNode = root.get("response");
            return responseNode != null ? responseNode.asText() : "ERROR: No response from model";

        } catch (IOException | InterruptedException e) {
            log.error("Ollama generate failed: {}", e.getMessage());
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    // ── GETTERS ───────────────────────────────────────────────────────

    public String getEmbedModel() { return embedModel; }
    public String getGenModel()   { return genModel; }
}
