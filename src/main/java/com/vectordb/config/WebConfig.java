package com.vectordb.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.*;

/**
 * CORS configuration — allows the frontend to call the API from any origin.
 *
 * C++ equivalent: void cors(httplib::Response& res) called in every handler.
 * Java: configure once globally, applies to all endpoints automatically.
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
            .allowedOrigins("*")
            .allowedMethods("GET", "POST", "DELETE", "OPTIONS")
            .allowedHeaders("*");
    }

    /**
     * Serve index.html (the frontend) from the classpath.
     * C++ equivalent: svr.Get("/", [](Request&, Response& res) { ... read index.html ... })
     */
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/", "/index.html")
            .addResourceLocations("classpath:/static/");
    }
}
