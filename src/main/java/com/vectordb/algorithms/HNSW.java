package com.vectordb.algorithms;

import com.vectordb.model.VectorItem;

import java.util.*;

/**
 * HNSW — Hierarchical Navigable Small World graph.
 *
 * This is the most important algorithm in modern vector databases.
 * Used by: Pinecone, Weaviate, Chroma, Milvus, Qdrant.
 *
 * THE CORE IDEA:
 *   Build a multilayer graph where:
 *   - Layer 0  has ALL nodes with many connections (dense graph)
 *   - Layer 1  has FEWER nodes with longer-range connections
 *   - Layer 2+ has even fewer nodes (exponentially fewer per layer)
 *
 *   SEARCH: Start at the top layer (fast highway), greedily navigate toward
 *   the query, drop down a layer, repeat. At layer 0, do a thorough local search.
 *   This gives O(log N) complexity.
 *
 * KEY PARAMETERS:
 *   M         = max connections per node per layer (default 16)
 *   M0        = max connections at layer 0 (= 2*M, denser base)
 *   ef_build  = beam width during insertion (higher = better quality, slower build)
 *   mL        = level multiplier = 1/ln(M), controls layer assignment probability
 *
 * C++ class HNSW { struct Node { VectorItem item; int maxLyr; vector<vector<int>> nbrs; }; ... }
 */
public class HNSW {

    /**
     * Node in the HNSW graph.
     *
     * C++ struct:
     *   struct Node {
     *       VectorItem item;
     *       int maxLyr;
     *       vector<vector<int>> nbrs;   // nbrs[layer] = list of neighbor IDs
     *   }
     *
     * Java: inner class with same fields.
     * nbrs is a List of Lists — nbrs.get(layer) returns neighbor IDs at that layer.
     */
    private static class Node {
        VectorItem item;
        int maxLayer;
        List<List<Integer>> neighbors;  // neighbors.get(layer) = neighbor IDs at that layer

        Node(VectorItem item, int maxLayer) {
            this.item      = item;
            this.maxLayer  = maxLayer;
            this.neighbors = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++) {
                neighbors.add(new ArrayList<>());
            }
        }
    }

    // The graph: maps node ID → Node
    // C++: std::unordered_map<int, Node> G
    // Java: HashMap<Integer, Node> G
    private final Map<Integer, Node> graph = new HashMap<>();

    private final int M;           // max connections per layer
    private final int M0;          // max connections at layer 0
    private final int efBuild;     // beam width during construction
    private final double mL;       // level multiplier = 1 / ln(M)

    private int topLayer = -1;     // highest layer in the graph
    private int entryPoint = -1;   // entry node ID (top of the graph)

    private final Random random = new Random(42);  // fixed seed for reproducibility

    public HNSW(int m, int efBuild) {
        this.M        = m;
        this.M0       = 2 * m;
        this.efBuild  = efBuild;
        this.mL       = 1.0 / Math.log(m);  // C++: mL(1.0f / std::log((float)m))
    }

    public HNSW() {
        this(16, 200);  // defaults from original C++ code
    }

    // ── LEVEL ASSIGNMENT ─────────────────────────────────────────────

    /**
     * Randomly assign a max layer to a new node.
     * Most nodes get layer 0; exponentially fewer get higher layers.
     * This creates the "small world" structure.
     *
     * C++ equivalent:
     *   int randLevel() {
     *       uniform_real_distribution<float> u(0.0f, 1.0f);
     *       return (int)floor(-log(u(rng)) * mL);
     *   }
     */
    private int randomLevel() {
        return (int) Math.floor(-Math.log(random.nextDouble()) * mL);
    }

    // ── INSERT ────────────────────────────────────────────────────────

    /**
     * Insert a vector into the HNSW graph.
     *
     * Algorithm:
     *   1. Assign a random max layer to the new node
     *   2. From the entry point, greedy-descend from topLayer to (newNode's layer + 1)
     *      keeping only 1 nearest candidate (fast coarse navigation)
     *   3. From newNode's layer down to 0:
     *      a. Run beam search with ef=efBuild candidates
     *      b. Connect new node to the M nearest found
     *      c. For each neighbor, add back-edge and prune if needed
     *   4. Update entry point if new node has a higher layer
     */
    public void insert(VectorItem item, DistanceMetric metric) {
        int id    = item.getId();
        int level = randomLevel();

        graph.put(id, new Node(item, level));

        if (entryPoint == -1) {
            // First node ever inserted — becomes the sole entry point
            entryPoint = id;
            topLayer   = level;
            return;
        }

        int ep = entryPoint;

        // Phase 1: Greedy descent from topLayer to level+1 (coarse navigation)
        // We only keep 1 candidate here — just finding the neighborhood
        for (int lc = topLayer; lc > level; lc--) {
            List<SearchCandidate> w = searchLayer(item.getEmbedding(), ep, 1, lc, metric);
            if (!w.isEmpty()) ep = w.get(0).id;
        }

        // Phase 2: Proper beam search from min(topLayer,level) down to 0
        for (int lc = Math.min(topLayer, level); lc >= 0; lc--) {
            List<SearchCandidate> w = searchLayer(item.getEmbedding(), ep, efBuild, lc, metric);
            int maxM = (lc == 0) ? M0 : M;

            // Select the M nearest as neighbors for this new node
            List<Integer> selected = selectNeighbors(w, maxM);
            graph.get(id).neighbors.get(lc).addAll(selected);

            // Add bidirectional edges: for each chosen neighbor, add id as their neighbor too
            for (int neighborId : selected) {
                Node neighborNode = graph.get(neighborId);
                if (neighborNode == null) continue;

                // Ensure the neighbor has a neighbor list for this layer
                while (neighborNode.neighbors.size() <= lc) {
                    neighborNode.neighbors.add(new ArrayList<>());
                }

                List<Integer> conn = neighborNode.neighbors.get(lc);
                conn.add(id);

                // Prune neighbor's connections if it now has too many
                if (conn.size() > maxM) {
                    // Rerank all connections by distance, keep only maxM closest
                    List<SearchCandidate> ds = new ArrayList<>();
                    for (int cId : conn) {
                        Node cNode = graph.get(cId);
                        if (cNode != null) {
                            float d = metric.compute(neighborNode.item.getEmbedding(),
                                                     cNode.item.getEmbedding());
                            ds.add(new SearchCandidate(d, cId));
                        }
                    }
                    ds.sort(Comparator.comparingDouble(c -> c.distance));
                    conn.clear();
                    for (int i = 0; i < Math.min(maxM, ds.size()); i++) {
                        conn.add(ds.get(i).id);
                    }
                }
            }

            if (!w.isEmpty()) ep = w.get(0).id;
        }

        // Update the global entry point if this node reaches a higher layer
        if (level > topLayer) {
            topLayer   = level;
            entryPoint = id;
        }
    }

    // ── SEARCH LAYER ─────────────────────────────────────────────────

    /**
     * Search one layer of the graph using a beam search (greedy best-first).
     *
     * Returns the ef nearest candidates to the query starting from entry point ep.
     *
     * C++ equivalent:
     *   vector<pair<float,int>> searchLayer(vector<float>& q, int ep, int ef, int lyr, DistFn dist)
     *
     * Uses two priority queues:
     *   cands = min-heap of candidates to explore (C++: greater<> comparator)
     *   found = max-heap of best results found so far (C++: default)
     *
     * Java equivalents:
     *   cands: PriorityQueue<>(comparingDouble) — min-heap (natural order for SearchCandidate)
     *   found: PriorityQueue<>(reverseOrder())  — max-heap (peek = farthest found)
     */
    private List<SearchCandidate> searchLayer(List<Float> query, int ep, int ef,
                                               int layer, DistanceMetric metric) {
        Set<Integer> visited  = new HashSet<>();
        // Min-heap: we always expand the closest unexplored candidate
        PriorityQueue<SearchCandidate> candidates =
            new PriorityQueue<>(Comparator.comparingDouble(c -> c.distance));
        // Max-heap: peek() gives us the FARTHEST of our ef best results
        PriorityQueue<SearchCandidate> found =
            new PriorityQueue<>(Comparator.comparingDouble((SearchCandidate c) -> c.distance).reversed());

        float d0 = metric.compute(query, graph.get(ep).item.getEmbedding());
        visited.add(ep);
        candidates.offer(new SearchCandidate(d0, ep));
        found.offer(new SearchCandidate(d0, ep));

        while (!candidates.isEmpty()) {
            SearchCandidate current = candidates.poll();

            // Pruning condition: if current candidate is farther than our ef-th best result,
            // no future candidate can improve our result set — stop early
            if (found.size() >= ef && current.distance > found.peek().distance) break;

            // Expand neighbors of current node at this layer
            Node currentNode = graph.get(current.id);
            if (currentNode == null || layer >= currentNode.neighbors.size()) continue;

            for (int neighborId : currentNode.neighbors.get(layer)) {
                if (visited.contains(neighborId) || !graph.containsKey(neighborId)) continue;
                visited.add(neighborId);

                float nd = metric.compute(query, graph.get(neighborId).item.getEmbedding());

                // Add to found if better than our current ef-th best
                if (found.size() < ef || nd < found.peek().distance) {
                    candidates.offer(new SearchCandidate(nd, neighborId));
                    found.offer(new SearchCandidate(nd, neighborId));
                    if (found.size() > ef) found.poll();  // remove farthest
                }
            }
        }

        // Sort results by distance (closest first) before returning
        List<SearchCandidate> result = new ArrayList<>(found);
        result.sort(Comparator.comparingDouble(c -> c.distance));
        return result;
    }

    // ── KNN SEARCH ────────────────────────────────────────────────────

    /**
     * Find the K nearest neighbors to a query vector.
     *
     * C++ equivalent:
     *   vector<pair<float,int>> knn(const vector<float>& q, int k, int ef, DistFn dist)
     */
    public List<BruteForce.SearchResult> knn(List<Float> query, int k, int ef, DistanceMetric metric) {
        if (entryPoint == -1) return Collections.emptyList();

        int ep = entryPoint;

        // Coarse navigation: descend from top layer to layer 1 keeping 1 candidate
        for (int lc = topLayer; lc > 0; lc--) {
            List<SearchCandidate> w = searchLayer(query, ep, 1, lc, metric);
            if (!w.isEmpty()) ep = w.get(0).id;
        }

        // Final search at layer 0 with ef candidates
        List<SearchCandidate> w = searchLayer(query, ep, Math.max(ef, k), 0, metric);

        // Return top K as SearchResult objects
        List<BruteForce.SearchResult> results = new ArrayList<>();
        for (int i = 0; i < Math.min(k, w.size()); i++) {
            results.add(new BruteForce.SearchResult(w.get(i).distance, w.get(i).id));
        }
        return results;
    }

    // ── REMOVE ────────────────────────────────────────────────────────

    /**
     * Remove a node from the graph.
     * Simple approach: remove from graph and all neighbor lists.
     * (Production HNSW would do more sophisticated re-linking.)
     *
     * C++ equivalent: void remove(int id) { ... G.erase(id); }
     */
    public void remove(int id) {
        if (!graph.containsKey(id)) return;

        // Remove this id from every neighbor list in every node
        for (Node node : graph.values()) {
            for (List<Integer> layerNeighbors : node.neighbors) {
                layerNeighbors.removeIf(nId -> nId == id);
            }
        }

        // If we're removing the entry point, find a new one
        if (entryPoint == id) {
            entryPoint = -1;
            for (Integer nid : graph.keySet()) {
                if (nid != id) { entryPoint = nid; break; }
            }
        }

        graph.remove(id);
    }

    // ── GRAPH INFO ────────────────────────────────────────────────────

    /**
     * Returns detailed information about the HNSW graph structure.
     * Used by the /hnsw-info REST endpoint.
     */
    public GraphInfo getInfo() {
        GraphInfo info = new GraphInfo();
        info.topLayer  = topLayer;
        info.nodeCount = graph.size();

        int maxL = Math.max(topLayer + 1, 1);
        info.nodesPerLayer = new int[maxL];
        info.edgesPerLayer = new int[maxL];
        info.nodes = new ArrayList<>();
        info.edges = new ArrayList<>();

        for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
            int id    = entry.getKey();
            Node node = entry.getValue();

            info.nodes.add(new GraphInfo.NodeView(id, node.item.getMetadata(),
                                                   node.item.getCategory(), node.maxLayer));

            for (int lc = 0; lc <= node.maxLayer && lc < maxL; lc++) {
                info.nodesPerLayer[lc]++;
                if (lc < node.neighbors.size()) {
                    for (int neighborId : node.neighbors.get(lc)) {
                        if (id < neighborId) {  // count each edge once
                            info.edgesPerLayer[lc]++;
                            info.edges.add(new GraphInfo.EdgeView(id, neighborId, lc));
                        }
                    }
                }
            }
        }
        return info;
    }

    public int size() { return graph.size(); }

    // ── HELPER CLASSES ────────────────────────────────────────────────

    /** Select the maxM nearest from a sorted candidate list. */
    private List<Integer> selectNeighbors(List<SearchCandidate> candidates, int maxM) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(maxM, candidates.size()); i++) {
            result.add(candidates.get(i).id);
        }
        return result;
    }

    /**
     * Internal search candidate — similar to C++ pair<float,int>.
     * Using a record (Java 16+) for concise immutable data.
     */
    private record SearchCandidate(float distance, int id) {}

    /**
     * Graph info returned by /hnsw-info endpoint.
     */
    public static class GraphInfo {
        public int topLayer, nodeCount;
        public int[] nodesPerLayer, edgesPerLayer;
        public List<NodeView> nodes;
        public List<EdgeView> edges;

        public record NodeView(int id, String metadata, String category, int maxLayer) {}
        public record EdgeView(int src, int dst, int layer) {}
    }
}
