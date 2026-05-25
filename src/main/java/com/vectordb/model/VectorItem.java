package com.vectordb.model;

import java.util.List;

/**
 * Represents a single vector stored in the database.
 *
 * C++ equivalent: struct VectorItem { int id; string metadata; string category; vector<float> emb; }
 *
 * In Java we use:
 *   - int         → int (primitive, same)
 *   - std::string → String (Java built-in)
 *   - vector<float> → List<Float> (Java generics can't use primitives, so Float not float)
 */
public class VectorItem {

    private int id;
    private String metadata;   // human-readable label e.g. "Binary Search Tree"
    private String category;   // e.g. "cs", "math", "food", "sports"
    private List<Float> embedding;  // the actual vector values

    // ── Constructors ──────────────────────────────────────────────────

    public VectorItem() {}

    public VectorItem(int id, String metadata, String category, List<Float> embedding) {
        this.id        = id;
        this.metadata  = metadata;
        this.category  = category;
        this.embedding = embedding;
    }

    // ── Getters & Setters ─────────────────────────────────────────────
    // (In a real project you'd use Lombok @Data, but we write them out
    //  manually so you can see every field clearly)

    public int getId()                  { return id; }
    public void setId(int id)           { this.id = id; }

    public String getMetadata()                  { return metadata; }
    public void setMetadata(String metadata)     { this.metadata = metadata; }

    public String getCategory()                  { return category; }
    public void setCategory(String category)     { this.category = category; }

    public List<Float> getEmbedding()                     { return embedding; }
    public void setEmbedding(List<Float> embedding)       { this.embedding = embedding; }

    @Override
    public String toString() {
        return "VectorItem{id=" + id + ", metadata='" + metadata + "', category='" + category + "'}";
    }
}
