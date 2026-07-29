package com.vectordb.algorithms;

import com.vectordb.model.VectorItem;

import java.util.*;

/**
 * BRUTE FORCE search — the simplest possible K-Nearest Neighbors (KNN) algorithm.
 *
 * How it works:
 *   1. Compute distance from query to EVERY vector in the database
 *   2. Sort all distances
 *   3. Return the K smallest (nearest) ones
 *
 * Time complexity: O(N·D) where N = number of vectors, D = dimensions
 * This is the "baseline" — always returns exact results, but slow for large N.
 *
 * C++ equivalent: class BruteForce { ... }
 *
 * Key Java vs C++ differences:
 *   - C++ uses vector<VectorItem>, Java uses ArrayList<VectorItem>
 *   - C++ uses pair<float,int>, Java uses a simple SearchResult record
 *   - C++ uses lambda captures, Java uses method references or lambdas
 *   - No manual memory management in Java (no new/delete)
 */
public class BruteForce {

    // In C++: std::vector<VectorItem> items;
    // In Java: ArrayList gives us dynamic resizing, just like std::vector
    private final List<VectorItem> items = new ArrayList<>();

    /**
     * Insert a vector into the brute-force index.
     * C++ equivalent: void insert(const VectorItem& v) { items.push_back(v); }
     */
    public void insert(VectorItem item) {
        items.add(item);
    }

    /**
     * K-Nearest Neighbor search.
     * Returns the K closest vectors to the query, sorted by distance (closest first).
     *
     * C++ equivalent:
     *   vector<pair<float,int>> knn(const vector<float>& q, int k, DistFn dist) {
     *       ...sort and resize...
     *   }
     *
     * Java equivalent uses List<SearchResult> instead of vector<pair<float,int>>
     */
    public List<SearchResult> knn(List<Float> query, int k, DistanceMetric metric) {
        List<SearchResult> results = new ArrayList<>(items.size());

        // Compute distance from query to every stored vector
        for (VectorItem item : items) {
            float dist = metric.compute(query, item.getEmbedding());
            results.add(new SearchResult(dist, item.getId()));
        }

        // Sort by distance ascending (nearest first)
        // C++: std::sort(r.begin(), r.end())  — pairs sort by first element by default
        // Java: Comparator.comparingDouble extracts the float for sorting
        results.sort(Comparator.comparingDouble(SearchResult::distance));

        // Return only the top K results
        return results.subList(0, Math.min(k, results.size()));
    }

    /**
     * Remove a vector by its ID.
     *
     * C++ equivalent:
     *   items.erase(remove_if(items.begin(), items.end(),
     *       [id](const VectorItem& v){ return v.id == id; }), items.end());
     *
     * Java equivalent uses removeIf with a lambda predicate.
     */
    public void remove(int id) {
        items.removeIf(item -> item.getId() == id);
    }

    /**
     * Clear all items (used when rebuilding after a delete).
     */
    public void rebuild(List<VectorItem> newItems) {
        items.clear();
        items.addAll(newItems);
    }

    public int size() { return items.size(); }

    /**
     * SearchResult — Java equivalent of C++ std::pair<float, int>
     *
     * In C++ you'd use:  pair<float, int> where .first = distance, .second = id
     * In Java, we use a record (Java 16+) — a concise immutable data class.
     *
     * record = automatically generates constructor, getters, equals, hashCode, toString
     */
    public record SearchResult(float distance, int id) implements Comparable<SearchResult> {
        @Override
        public int compareTo(SearchResult other) {
            return Float.compare(this.distance, other.distance);
        }
    }
}
