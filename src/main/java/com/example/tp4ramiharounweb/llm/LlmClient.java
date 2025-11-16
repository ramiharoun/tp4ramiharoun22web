package com.example.tp4ramiharounweb.llm;

import jakarta.enterprise.context.Dependent;
import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;

import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;

@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private final ChatMemory chatMemory;
    private final GoogleAiGeminiChatModel model;

    // Assistant utilisé pour décider "RAG ou pas" (Test 4)
    private final RouterAssistant routerAssistant;

    // Test 4 : RAG conditionnel
    private final Assistant assistantTest4;

    // Test 5 : PDF + Web Tavily
    private final Assistant assistantTest5;

    public LlmClient() {

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY manquante !");
        }

        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // -------------- Ingestion des 2 PDF --------------
        ClassLoader cl = Thread.currentThread().getContextClassLoader();

        URL url1 = cl.getResource("support_rag.pdf");
        if (url1 == null) throw new IllegalStateException("support_rag.pdf introuvable !");


        Path pdf1;
        Path pdf2;
        try {
            pdf1 = Path.of(url1.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Erreur URI PDF", e);
        }

        Document doc1 = FileSystemDocumentLoader.loadDocument(pdf1);

        var splitter = DocumentSplitters.recursive(500, 50);

        List<TextSegment> segments = new ArrayList<>();
        segments.addAll(splitter.split(doc1));


        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> vectors = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, segments);

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .maxResults(5)
                .build();

        // -------------- Assistant spécial pour le routage (Test 4) --------------
        this.routerAssistant = AiServices.builder(RouterAssistant.class)
                .chatModel(model)
                .build();

        // -------------- TEST 4 : RAG conditionnel --------------
        QueryRouter conditionalRouter = new ConditionalRagRouter(pdfRetriever);

        RetrievalAugmentor augmentorTest4 = DefaultRetrievalAugmentor.builder()
                .queryRouter(conditionalRouter)
                .build();

        this.assistantTest4 = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentorTest4)
                .build();

        // -------------- TEST 5 : PDF + WEB TAVILY --------------
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            throw new IllegalStateException("Variable d'environnement TAVILY_API_KEY manquante !");
        }

        TavilyWebSearchEngine webEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webEngine)
                .maxResults(5)
                .build();

        DefaultQueryRouter routerTest5 = new DefaultQueryRouter(pdfRetriever, webRetriever);

        RetrievalAugmentor augmentorTest5 = DefaultRetrievalAugmentor.builder()
                .queryRouter(routerTest5)
                .build();

        this.assistantTest5 = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentorTest5)
                .build();
    }

    // Interface interne pour interroger le LM dans le QueryRouter (Test 4)
    private interface RouterAssistant {
        String decide(String question);
    }

    // ------------------- TEST 4 : ROUTAGE CONDITIONNEL -------------------
    private class ConditionalRagRouter implements QueryRouter {

        private final ContentRetriever retriever;

        ConditionalRagRouter(ContentRetriever retriever) {
            this.retriever = retriever;
        }

        @Override
        public List<ContentRetriever> route(Query query) {

            String question = query.text();

            String prompt = """
        Est-ce que la requête suivante porte sur l'intelligence artificielle (IA)
        ou sur le RAG (Retrieval Augmented Generation) ?
        "%s"
        Réponds seulement par : oui, non ou peut-être.
        """.formatted(question);

            String decision = routerAssistant.decide(prompt);
            if (decision == null) return List.of();

            String normalized = decision.toLowerCase(Locale.FRENCH).trim();

            if (normalized.startsWith("non")) {
                return List.of(); // pas de RAG
            }

            return List.of(retriever); // oui ou peut-être
        }
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = (systemRole == null || systemRole.isBlank())
                ? "You are a helpful assistant."
                : systemRole;

        chatMemory.clear();
        chatMemory.add(SystemMessage.from(this.systemRole));
    }

    // ------------------- Méthodes TEST 4 -------------------
    public String chatTest4(String prompt) {
        if (prompt == null || prompt.isBlank()) return "";
        return assistantTest4.chat(prompt.trim());
    }

    public String chatTest4(String systemRole, String prompt) {
        setSystemRole(systemRole);
        return chatTest4(prompt);
    }

    // ------------------- Méthodes TEST 5 -------------------
    public String chatTest5(String prompt) {
        if (prompt == null || prompt.isBlank()) return "";
        return assistantTest5.chat(prompt.trim());
    }

    public String chatTest5(String systemRole, String prompt) {
        setSystemRole(systemRole);
        return chatTest5(prompt);
    }

    public String chat(String prompt) {
        return chatTest5(prompt);
    }

    public String chat(String systemRole, String prompt) {
        setSystemRole(systemRole);
        return chat(prompt);
    }

    public String getSystemRole() {
        return systemRole;
    }
}