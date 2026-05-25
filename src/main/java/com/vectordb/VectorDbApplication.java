package com.vectordb;

import com.vectordb.service.OllamaClient;
import com.vectordb.service.VectorDBService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

/**
 * VectorDB Java — Spring Boot Application Entry Point
 *
 * C++ equivalent: int main() { ... svr.listen("0.0.0.0", 8080); }
 *
 * In Java/Spring Boot:
 *   - SpringApplication.run() starts the embedded Tomcat server
 *   - All @Service and @RestController beans are automatically detected
 *   - No need for manual wiring — Spring handles dependency injection
 *   - Server port is configured in application.properties
 *
 * @SpringBootApplication is a convenience annotation that combines:
 *   @Configuration + @EnableAutoConfiguration + @ComponentScan
 */
@SpringBootApplication
public class VectorDbApplication {

    public static void main(String[] args) {
        ApplicationContext ctx = SpringApplication.run(VectorDbApplication.class, args);

        // Startup banner — equivalent to C++ cout << "=== VectorDB Engine ===" << endl
        VectorDBService vectorDB = ctx.getBean(VectorDBService.class);
        OllamaClient    ollama   = ctx.getBean(OllamaClient.class);

        System.out.println();
        System.out.println("=== VectorDB Java Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(vectorDB.size() + " demo vectors | " + vectorDB.getDims()
                           + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollama.isAvailable()
            ? "ONLINE  embed=" + ollama.getEmbedModel() + "  gen=" + ollama.getGenModel()
            : "OFFLINE (install from ollama.com)"));
        System.out.println("Open your browser: http://localhost:8080");
        System.out.println();
    }
}
