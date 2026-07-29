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

 
    public List<SearchResult> knn(List<Float> query, int k, DistanceMetric metric) {
        List<SearchResult> results = new ArrayList<>(items.size());

        // Compute distance from query to every stored vector
        for (VectorItem item : items) {
            float dist = metric.compute(query, item.getEmbedding());
            results.add(new SearchResult(dist, item.getId()));
        }

      
        results.sort(Comparator.comparingDouble(SearchResult::distance));

        // Return only the top K results
        return results.subList(0, Math.min(k, results.size()));
    }


    public void remove(int id) {
        items.removeIf(item -> item.getId() == id);
    }

    public void rebuild(List<VectorItem> newItems) {
        items.clear();
        items.addAll(newItems);
    }

    public int size() { return items.size(); }

  
    public record SearchResult(float distance, int id) implements Comparable<SearchResult> {
        @Override
        public int compareTo(SearchResult other) {
            return Float.compare(this.distance, other.distance);
        }
    }
}
