package com.vectordb.algorithms;

import com.vectordb.model.VectorItem;

import java.util.*;

/**
 * K-DIMENSIONAL TREE (KD-Tree) — a space-partitioning data structure for KNN search.
 *
 * HOW IT WORKS:
 *   Insert: Build a binary tree by alternating which dimension to split on.
 *     - Level 0: split on dimension 0 (x-axis)
 *     - Level 1: split on dimension 1 (y-axis)
 *     - Level 2: split on dimension 2 (z-axis)
 *     - Level k: split on dimension (k % numDimensions)
 *
 *   Search: Use the tree structure to PRUNE branches that can't contain closer points.
 *     - This gives O(log N) average case — much faster than O(N) brute force.
 *     - BUT: degrades at high dimensions (768D) because pruning becomes ineffective.
 *
 * WHY KDTREE FAILS AT HIGH DIMENSIONS:
 *   In high-dimensional space, the distance to the hyperplane boundary is almost never
 *   large enough to prune the far branch. Almost every search visits nearly all nodes.
 *   This is called the "curse of dimensionality". HNSW doesn't have this problem.
 *
 * C++ equivalent:
 *   struct KDNode { VectorItem item; KDNode* left; KDNode* right; }
 *   class KDTree { KDNode* root; int dims; ... }
 *
 * Java differences:
 *   - No pointers: use object references (KDNode left, right)
 *   - No manual memory management: GC handles deallocation
 *   - Inner class instead of struct
 */
public class KDTree {

    /**
     * KDNode — one node in the binary tree.
     *
     * C++ struct: struct KDNode { VectorItem item; KDNode* left; KDNode* right; }
     * Java: private inner class (same concept, no pointers)
     */
    private static class KDNode {
        VectorItem item;
        KDNode left;
        KDNode right;

        KDNode(VectorItem item) {
            this.item = item;
        }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    // ── INSERT ────────────────────────────────────────────────────────

    /**
     * Insert a vector into the KD-Tree.
     *
     * C++ equivalent:
     *   KDNode* ins(KDNode* n, const VectorItem& v, int d) {
     *       if (!n) return new KDNode(v);
     *       int ax = d % dims;
     *       if (v.emb[ax] < n->item.emb[ax]) n->left = ins(n->left, v, d+1);
     *       else n->right = ins(n->right, v, d+1);
     *       return n;
     *   }
     *
     * Java uses the same recursive logic, but returns the node reference instead of a pointer.
     */
    public void insert(VectorItem item) {
        root = insertRecursive(root, item, 0);
    }

    private KDNode insertRecursive(KDNode node, VectorItem item, int depth) {
        if (node == null) return new KDNode(item);   // leaf position found

        int axis = depth % dims;   // which dimension to split on at this level

        // Go left if smaller, right if greater or equal — same as C++
        if (item.getEmbedding().get(axis) < node.item.getEmbedding().get(axis)) {
            node.left  = insertRecursive(node.left,  item, depth + 1);
        } else {
            node.right = insertRecursive(node.right, item, depth + 1);
        }
        return node;
    }

    // ── KNN SEARCH ────────────────────────────────────────────────────

    /**
     * K-Nearest Neighbor search using tree pruning.
     *
     * Uses a MAX-HEAP (PriorityQueue with reverse order) to track the K nearest
     * candidates seen so far. We prune a branch when its minimum possible distance
     * is already larger than our K-th best candidate.
     *
     * C++ uses: priority_queue<pair<float,int>> heap  (max-heap by default)
     * Java uses: PriorityQueue with Comparator.reverseOrder() for max-heap behavior
     */
    public List<BruteForce.SearchResult> knn(List<Float> query, int k, DistanceMetric metric) {
        // Max-heap: the TOP element is the FARTHEST of the K candidates
        // When we find something closer, we pop the farthest and add the new one
        PriorityQueue<BruteForce.SearchResult> heap =
            new PriorityQueue<>(k, Comparator.reverseOrder());  // max-heap

        knnRecursive(root, query, k, 0, metric, heap);

        // Convert heap to sorted list (closest first)
        List<BruteForce.SearchResult> result = new ArrayList<>(heap);
        result.sort(Comparator.naturalOrder());
        return result;
    }

    private void knnRecursive(KDNode node, List<Float> query, int k, int depth,
                               DistanceMetric metric, PriorityQueue<BruteForce.SearchResult> heap) {
        if (node == null) return;

        // Compute distance from query to this node
        float dist = metric.compute(query, node.item.getEmbedding());

        // Add to heap if we haven't found K yet, or if this is closer than the farthest known
        if (heap.size() < k || dist < heap.peek().distance()) {
            heap.offer(new BruteForce.SearchResult(dist, node.item.getId()));
            if (heap.size() > k) heap.poll();   // remove farthest if over limit
        }

        // Decide which subtree to search first (the "closer" side)
        int axis = depth % dims;
        float diff = query.get(axis) - node.item.getEmbedding().get(axis);
        KDNode closer  = diff < 0 ? node.left  : node.right;
        KDNode farther = diff < 0 ? node.right : node.left;

        // Always search the closer side
        knnRecursive(closer, query, k, depth + 1, metric, heap);

        // Only search the farther side if:
        //   (a) we haven't found K results yet, OR
        //   (b) the closest possible point in that subtree could beat our K-th best
        // The closest point in the farther subtree is AT LEAST |diff| away on this axis.
        // C++ equivalent: if (heap.size() < k || abs(diff) < heap.top().first)
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance()) {
            knnRecursive(farther, query, k, depth + 1, metric, heap);
        }
    }

    // ── REBUILD ───────────────────────────────────────────────────────

    /**
     * Rebuild the tree from scratch with new items.
     * Needed after deletion (KD-Trees don't support efficient single-node deletion).
     *
     * C++ equivalent: void rebuild(const vector<VectorItem>& items) { ... }
     */
    public void rebuild(List<VectorItem> items) {
        root = null;                    // C++: destroy(root); root = nullptr;
        for (VectorItem item : items) { // GC handles the old nodes
            insert(item);
        }
    }
}
