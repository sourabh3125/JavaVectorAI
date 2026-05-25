package com.vectordb.service;

import com.vectordb.algorithms.*;
import com.vectordb.model.VectorItem;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * VectorDB — the main in-memory vector database.
 *
 * Maintains THREE indexes simultaneously:
 *   1. BruteForce — O(N·D) exact search, always correct
 *   2. KDTree     — O(log N) exact search, degrades at high dimensions
 *   3. HNSW       — O(log N) approximate search, works at any dimension
 *
 * All three are kept in sync on every insert/delete.
 * The /benchmark endpoint runs all three and compares speed.
 *
 * Thread safety:
 *   C++ uses std::mutex + std::lock_guard
 *   Java uses ReentrantLock (more flexible than synchronized)
 *
 * Spring @Service annotation = this class is a Spring-managed singleton bean.
 * @PostConstruct = runs loadDemo() right after Spring wires all dependencies.
 */
@Service
public class VectorDBService {

    @Value("${vectordb.demo-dims:16}")
    private int dims;

    // Primary store: id → VectorItem  (C++: unordered_map<int, VectorItem> store)
    // ConcurrentHashMap is thread-safe for reads; we still lock for write operations
    private final Map<Integer, VectorItem> store = new ConcurrentHashMap<>();

    // The three indexes
    private BruteForce bruteForce;
    private KDTree     kdTree;
    private HNSW       hnsw;

    // Mutex for write operations (C++: std::mutex mu)
    // ReentrantLock allows the same thread to lock multiple times (safe for nested calls)
    private final ReentrantLock lock = new ReentrantLock();

    private int nextId = 1;

    @PostConstruct
    public void init() {
        bruteForce = new BruteForce();
        kdTree     = new KDTree(dims);
        hnsw       = new HNSW(16, 200);
        loadDemo();
    }

    // ── INSERT ────────────────────────────────────────────────────────

    /**
     * Insert a vector into all three indexes.
     *
     * C++ equivalent:
     *   int insert(const string& meta, const string& cat,
     *              const vector<float>& emb, DistFn dist)
     *   {
     *       lock_guard<mutex> lk(mu);
     *       VectorItem v{nextId++, meta, cat, emb};
     *       store[v.id] = v; bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
     *       return v.id;
     *   }
     */
    public int insert(String metadata, String category, List<Float> embedding) {
        lock.lock();
        try {
            VectorItem item = new VectorItem(nextId++, metadata, category, embedding);
            store.put(item.getId(), item);
            bruteForce.insert(item);
            kdTree.insert(item);
            hnsw.insert(item, DistanceMetric.cosine());
            return item.getId();
        } finally {
            lock.unlock();  // always unlock even if exception occurs
        }
    }

    // ── DELETE ────────────────────────────────────────────────────────

    /**
     * Delete a vector from all three indexes.
     * KDTree doesn't support efficient deletion, so we rebuild it from scratch.
     */
    public boolean delete(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bruteForce.remove(id);
            hnsw.remove(id);
            // Rebuild KDTree (C++: kdt.rebuild(rem))
            kdTree.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally {
            lock.unlock();
        }
    }

    // ── SEARCH ────────────────────────────────────────────────────────

    /**
     * Search using the specified algorithm and metric.
     * Returns results with timing information.
     *
     * C++ equivalent:
     *   SearchOut search(vector<float>& q, int k, string metric, string algo)
     */
    public SearchResult search(List<Float> query, int k, String metric, String algo) {
        lock.lock();
        try {
            DistanceMetric distFn = DistanceMetric.of(metric);
            long startNs = System.nanoTime();

            List<BruteForce.SearchResult> raw = switch (algo.toLowerCase()) {
                case "bruteforce" -> bruteForce.knn(query, k, distFn);
                case "kdtree"     -> kdTree.knn(query, k, distFn);
                default           -> hnsw.knn(query, k, 50, distFn);  // hnsw
            };

            long elapsedUs = (System.nanoTime() - startNs) / 1000;  // nanoseconds → microseconds

            // Enrich raw results with full VectorItem data
            List<Hit> hits = new ArrayList<>();
            for (BruteForce.SearchResult r : raw) {
                VectorItem item = store.get(r.id());
                if (item != null) {
                    hits.add(new Hit(item.getId(), item.getMetadata(), item.getCategory(),
                                     item.getEmbedding(), r.distance()));
                }
            }

            return new SearchResult(hits, elapsedUs, algo, metric);
        } finally {
            lock.unlock();
        }
    }

    // ── BENCHMARK ────────────────────────────────────────────────────

    /**
     * Run all three algorithms and return timing comparison.
     *
     * C++ equivalent:
     *   BenchOut benchmark(vector<float>& q, int k, string metric)
     */
    public BenchmarkResult benchmark(List<Float> query, int k, String metric) {
        lock.lock();
        try {
            DistanceMetric distFn = DistanceMetric.of(metric);

            long t0 = System.nanoTime();
            bruteForce.knn(query, k, distFn);
            long bfUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            kdTree.knn(query, k, distFn);
            long kdUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            hnsw.knn(query, k, 50, distFn);
            long hnswUs = (System.nanoTime() - t0) / 1000;

            return new BenchmarkResult(bfUs, kdUs, hnswUs, store.size());
        } finally {
            lock.unlock();
        }
    }

    // ── ACCESSORS ────────────────────────────────────────────────────

    public List<VectorItem> getAll() {
        return new ArrayList<>(store.values());
    }

    public HNSW.GraphInfo getHnswInfo() {
        lock.lock();
        try { return hnsw.getInfo(); }
        finally { lock.unlock(); }
    }

    public int size() { return store.size(); }
    public int getDims() { return dims; }

    // ── RESULT TYPES ─────────────────────────────────────────────────
    // Java records — C++ equivalent of structs used as return types

    public record Hit(int id, String metadata, String category,
                      List<Float> embedding, float distance) {}

    public record SearchResult(List<Hit> hits, long latencyUs,
                               String algo, String metric) {}

    public record BenchmarkResult(long bruteforceUs, long kdtreeUs,
                                  long hnswUs, int itemCount) {}

    // ── DEMO DATA ────────────────────────────────────────────────────

    /**
     * Pre-load 20 demo vectors: 5 CS, 5 Math, 5 Food, 5 Sports.
     * Each is a 16D vector where meaningful dimensions are:
     *   dims 0-3:  CS concepts
     *   dims 4-7:  Math concepts
     *   dims 8-11: Food concepts
     *   dims 12-15: Sports concepts
     *
     * C++ equivalent: void loadDemo(VectorDB& db) { ... }
     */
    private void loadDemo() {
        // CS
        insert("Linked List: nodes connected by pointers", "cs",
            Arrays.asList(0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f));
        insert("Binary Search Tree: O(log n) search and insert", "cs",
            Arrays.asList(0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f));
        insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            Arrays.asList(0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f));
        insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            Arrays.asList(0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f));
        insert("Hash Table: O(1) lookup with collision chaining", "cs",
            Arrays.asList(0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f));
        // Math
        insert("Calculus: derivatives integrals and limits", "math",
            Arrays.asList(0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f));
        insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            Arrays.asList(0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f));
        insert("Probability: distributions random variables Bayes theorem", "math",
            Arrays.asList(0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f));
        insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            Arrays.asList(0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f));
        insert("Combinatorics: permutations combinations generating functions", "math",
            Arrays.asList(0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f));
        // Food
        insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            Arrays.asList(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f));
        insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            Arrays.asList(0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f));
        insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            Arrays.asList(0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f));
        insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            Arrays.asList(0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f));
        insert("Croissant: laminated pastry with buttery flaky layers", "food",
            Arrays.asList(0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f));
        // Sports
        insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            Arrays.asList(0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f));
        insert("Football: tackles touchdowns field goals and strategy", "sports",
            Arrays.asList(0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f));
        insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            Arrays.asList(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f));
        insert("Chess: openings endgames tactics strategic board game", "sports",
            Arrays.asList(0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f));
        insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            Arrays.asList(0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f));
    }
}
