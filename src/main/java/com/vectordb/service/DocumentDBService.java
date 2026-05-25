package com.vectordb.service;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.DistanceMetric;
import com.vectordb.algorithms.HNSW;
import com.vectordb.model.DocItem;
import com.vectordb.model.VectorItem;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * DocumentDB — stores real document chunks with Ollama-generated embeddings.
 *
 * This powers the RAG (Retrieval Augmented Generation) pipeline:
 *   1. User pastes a document → we chunk it into 250-word pieces
 *   2. Each chunk is sent to Ollama → converted to a 768D vector
 *   3. Vectors are stored in HNSW for fast retrieval
 *   4. At query time: embed the question → find similar chunks → send to LLM
 *
 * C++ equivalent: class DocumentDB { ... HNSW hnsw; BruteForce bf; ... }
 *
 * Uses HNSW for large collections (10+ docs), BruteForce for small ones
 * because HNSW needs some nodes to be useful.
 */
@Service
public class DocumentDBService {

    @Value("${vectordb.chunk-words:250}")
    private int chunkWords;

    @Value("${vectordb.chunk-overlap:30}")
    private int chunkOverlap;

    private final Map<Integer, DocItem> store = new ConcurrentHashMap<>();
    private final ReentrantLock lock          = new ReentrantLock();

    private HNSW       hnsw       = new HNSW(16, 200);
    private BruteForce bruteForce = new BruteForce();

    private int nextId = 1;
    private int dims   = 0;  // determined from first inserted embedding

    private final OllamaClient ollamaClient;

    public DocumentDBService(OllamaClient ollamaClient) {
        this.ollamaClient = ollamaClient;
    }

    // ── INSERT ────────────────────────────────────────────────────────

    /**
     * Insert one document chunk with its pre-computed embedding.
     *
     * C++ equivalent:
     *   int insert(const string& title, const string& text, const vector<float>& emb)
     */
    public int insert(String title, String text, List<Float> embedding) {
        lock.lock();
        try {
            if (dims == 0) dims = embedding.size();

            DocItem doc = new DocItem(nextId++, title, text, embedding);
            store.put(doc.getId(), doc);

            // Store as VectorItem for algorithm compatibility
            VectorItem vi = new VectorItem(doc.getId(), title, "doc", embedding);
            hnsw.insert(vi, DistanceMetric.cosine());
            bruteForce.insert(vi);

            return doc.getId();
        } finally {
            lock.unlock();
        }
    }

    // ── EMBED AND INSERT DOCUMENT ────────────────────────────────────

    /**
     * Full pipeline: chunk text → embed each chunk → store all chunks.
     * Returns the IDs of all stored chunks.
     *
     * Throws IllegalStateException if Ollama is unavailable.
     */
    public List<Integer> insertDocument(String title, String text) {
        List<String> chunks = chunkText(text);
        List<Integer> ids   = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            List<Float> embedding = ollamaClient.embed(chunks.get(i));

            if (embedding.isEmpty()) {
                throw new IllegalStateException(
                    "Ollama unavailable. Install from https://ollama.com then run: " +
                    "ollama pull nomic-embed-text && ollama pull llama3.2"
                );
            }

            // Label chunk with position if document was split
            String chunkTitle = chunks.size() > 1
                ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                : title;

            ids.add(insert(chunkTitle, chunks.get(i), embedding));
        }

        return ids;
    }

    // ── SEARCH ────────────────────────────────────────────────────────

    /**
     * Find the K most semantically similar chunks to the query vector.
     *
     * C++ equivalent:
     *   vector<pair<float, DocItem>> search(const vector<float>& q, int k, float max_dist)
     */
    public List<DocSearchResult> search(List<Float> queryEmbedding, int k) {
        lock.lock();
        try {
            if (store.isEmpty()) return List.of();

            // Use BruteForce for small collections, HNSW for larger ones
            List<BruteForce.SearchResult> raw = (store.size() < 10)
                ? bruteForce.knn(queryEmbedding, k, DistanceMetric.cosine())
                : hnsw.knn(queryEmbedding, k, 50, DistanceMetric.cosine());

            List<DocSearchResult> results = new ArrayList<>();
            for (BruteForce.SearchResult r : raw) {
                DocItem doc = store.get(r.id());
                if (doc != null && r.distance() <= 0.7f) {  // filter by similarity threshold
                    results.add(new DocSearchResult(r.distance(), doc));
                }
            }
            return results;
        } finally {
            lock.unlock();
        }
    }

    // ── RAG PIPELINE ─────────────────────────────────────────────────

    /**
     * Full RAG (Retrieval Augmented Generation) pipeline:
     *   1. Embed the question
     *   2. Retrieve top-k relevant chunks
     *   3. Build a prompt combining question + context
     *   4. Send to LLM and return the answer
     *
     * C++ equivalent: /doc/ask endpoint handler in main()
     */
    public RagResult ask(String question, int k) {
        // Step 1: embed the question
        List<Float> questionEmbedding = ollamaClient.embed(question);
        if (questionEmbedding.isEmpty()) {
            throw new IllegalStateException("Ollama unavailable. Run: ollama serve");
        }

        // Step 2: retrieve relevant chunks
        List<DocSearchResult> hits = search(questionEmbedding, k);

        // Step 3: build RAG prompt
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            context.append("[").append(i + 1).append("] ")
                   .append(hits.get(i).doc().getTitle()).append(":\n")
                   .append(hits.get(i).doc().getText()).append("\n\n");
        }

        String prompt =
            "You are a helpful assistant. Answer the user's question directly. " +
            "Use the provided context if it contains relevant information. " +
            "If it doesn't, just use your own general knowledge. " +
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like " +
            "'the context doesn't mention'. Just answer the question naturally.\n\n" +
            "Context:\n" + context +
            "Question: " + question + "\n\nAnswer:";

        // Step 4: generate answer
        String answer = ollamaClient.generate(prompt);

        return new RagResult(answer, ollamaClient.getGenModel(), hits, store.size());
    }

    // ── TEXT CHUNKER ─────────────────────────────────────────────────

    /**
     * Split text into overlapping chunks of chunkWords words.
     * Overlap ensures context is not lost at chunk boundaries.
     *
     * C++ equivalent:
     *   vector<string> chunkText(const string& text, int chunkWords, int overlapWords)
     */
    private List<String> chunkText(String text) {
        // Split on whitespace — C++: istringstream to split into words
        String[] words = text.trim().split("\\s+");

        if (words.length <= chunkWords) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - chunkOverlap;  // how far to advance between chunks

        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(start + chunkWords, words.length);
            // Join words back into a string — C++: for loop with chunk += words[j]
            String chunk = String.join(" ", Arrays.copyOfRange(words, start, end));
            chunks.add(chunk);
            if (end == words.length) break;
        }

        return chunks;
    }

    // ── DELETE / LIST ─────────────────────────────────────────────────

    public boolean delete(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            // Rebuild brute force index
            BruteForce newBf = new BruteForce();
            for (DocItem doc : store.values()) {
                newBf.insert(new VectorItem(doc.getId(), doc.getTitle(), "doc", doc.getEmbedding()));
            }
            bruteForce = newBf;
            return true;
        } finally {
            lock.unlock();
        }
    }

    public List<DocItem> getAll() {
        return new ArrayList<>(store.values());
    }

    public int size()    { return store.size(); }
    public int getDims() { return dims; }

    // ── RESULT TYPES ─────────────────────────────────────────────────

    public record DocSearchResult(float distance, DocItem doc) {}

    public record RagResult(String answer, String model,
                            List<DocSearchResult> contexts, long docCount) {}
}
