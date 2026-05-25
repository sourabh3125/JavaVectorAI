package com.vectordb.model;

import java.util.List;

/**
 * Represents one chunk of a real document stored in DocumentDB.
 * Documents are split into overlapping chunks, each embedded by Ollama.
 *
 * C++ equivalent: struct DocItem { int id; string title; string text; vector<float> emb; }
 */
public class DocItem {

    private int id;
    private String title;      // e.g. "Operating Systems Notes [1/3]"
    private String text;       // the actual chunk text (up to 250 words)
    private List<Float> embedding;  // 768-dimensional vector from nomic-embed-text

    public DocItem() {}

    public DocItem(int id, String title, String text, List<Float> embedding) {
        this.id        = id;
        this.title     = title;
        this.text      = text;
        this.embedding = embedding;
    }

    public int getId()                     { return id; }
    public void setId(int id)              { this.id = id; }

    public String getTitle()               { return title; }
    public void setTitle(String title)     { this.title = title; }

    public String getText()                { return text; }
    public void setText(String text)       { this.text = text; }

    public List<Float> getEmbedding()                  { return embedding; }
    public void setEmbedding(List<Float> embedding)    { this.embedding = embedding; }
}
