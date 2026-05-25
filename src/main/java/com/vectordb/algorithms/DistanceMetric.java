package com.vectordb.algorithms;

import java.util.List;

/**
 * DistanceMetric is a functional interface — a Java equivalent of C++'s:
 *   using DistFn = std::function<float(const vector<float>&, const vector<float>&)>
 *
 * In Java, @FunctionalInterface means this interface has exactly ONE abstract method.
 * This lets us use lambda expressions or method references anywhere DistanceMetric is needed.
 *
 * Example usage:
 *   DistanceMetric metric = DistanceMetric.cosine();
 *   float distance = metric.compute(vectorA, vectorB);
 */
@FunctionalInterface
public interface DistanceMetric {

    /**
     * Compute the distance between two vectors.
     * Lower value = more similar (except for raw cosine similarity, but we use cosine *distance* = 1 - similarity).
     */
    float compute(List<Float> a, List<Float> b);

    // ── Static factory methods — returns the right metric by name ─────

    static DistanceMetric of(String metricName) {
        return switch (metricName.toLowerCase()) {
            case "cosine"    -> cosine();
            case "manhattan" -> manhattan();
            default          -> euclidean();
        };
    }

    /**
     * EUCLIDEAN distance — straight-line distance in N-dimensional space.
     * Formula: sqrt( sum( (a[i] - b[i])^2 ) )
     *
     * C++ equivalent:
     *   float euclidean(const vector<float>& a, const vector<float>& b) {
     *       float s = 0;
     *       for (int i = 0; i < a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
     *       return sqrt(s);
     *   }
     */
    static DistanceMetric euclidean() {
        return (a, b) -> {
            float sum = 0f;
            for (int i = 0; i < a.size(); i++) {
                float diff = a.get(i) - b.get(i);
                sum += diff * diff;
            }
            return (float) Math.sqrt(sum);
        };
    }

    /**
     * COSINE distance — measures angle between two vectors.
     * Formula: 1 - (dot(a,b) / (|a| * |b|))
     *
     * Returns 0 if vectors point in the same direction (most similar),
     * returns 1 if perpendicular, returns 2 if opposite.
     * Used most often for semantic similarity in NLP/AI.
     */
    static DistanceMetric cosine() {
        return (a, b) -> {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < a.size(); i++) {
                dot   += a.get(i) * b.get(i);
                normA += a.get(i) * a.get(i);
                normB += b.get(i) * b.get(i);
            }
            if (normA < 1e-9f || normB < 1e-9f) return 1.0f;  // zero vector guard
            return 1.0f - dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
        };
    }

    /**
     * MANHATTAN distance — sum of absolute differences (like city blocks).
     * Formula: sum( |a[i] - b[i]| )
     *
     * Faster to compute than Euclidean (no sqrt), useful for sparse data.
     */
    static DistanceMetric manhattan() {
        return (a, b) -> {
            float sum = 0f;
            for (int i = 0; i < a.size(); i++) {
                sum += Math.abs(a.get(i) - b.get(i));
            }
            return sum;
        };
    }
}
