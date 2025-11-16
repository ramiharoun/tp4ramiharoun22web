package com.example.tp4ramiharounweb.llm;

import jakarta.enterprise.context.Dependent;
import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.util.List;

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
import dev.langchain4j.rag.query.router.DefaultQueryRouter;

import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;

/**
 * Classe "métier" qui gère l'accès au LLM Gemini via LangChain4j.
 * Elle crée le modèle, la mémoire et l'assistant (proxy généré automatiquement).
 */
@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private final ChatMemory chatMemory;
    private final GoogleAiGeminiChatModel model;
    private final Assistant assistant;

    public LlmClient() {
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY manquante !");
        }

        // 1. Création du modèle Gemini
        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        // 2. Mémoire : conserve les 10 derniers messages
        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Test 5 : RAG avec PDF local + recherche Web Tavily


        URL url = Thread.currentThread().getContextClassLoader().getResource("support_rag.pdf");
        if (url == null) {
            throw new IllegalStateException("Ressource introuvable : support_rag.pdf");
        }

        Path pdfPath;
        try {
            pdfPath = Path.of(url.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("URI invalide pour support_rag.pdf", e);
        }

        Document doc = FileSystemDocumentLoader.loadDocument(pdfPath);
        var splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> vectors = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, segments);

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .maxResults(5)
                .build();

        // Recherche Web : Tavily
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            throw new IllegalStateException("Variable d'environnement TAVILY_API_KEY manquante !");
        }

        TavilyWebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(5)
                .build();

        // Routage par défaut : utilise PDF + Web
        DefaultQueryRouter router = new DefaultQueryRouter(pdfRetriever, webRetriever);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    /** Définit le rôle système et réinitialise la mémoire. */
    public void setSystemRole(String systemRole) {
        this.systemRole = (systemRole == null || systemRole.isBlank())
                ? "You are a helpful assistant."
                : systemRole;

        chatMemory.clear();
        chatMemory.add(SystemMessage.from(this.systemRole));
    }

    /** Envoie une requête au LLM (le rôle système est déjà défini). */
    public String chat(String prompt) {
        if (prompt == null || prompt.isBlank()) return "";
        return assistant.chat(prompt.trim());
    }

    /** Variante : permet de définir le rôle et d’envoyer la question en un seul appel. */
    public String chat(String systemRole, String prompt) {
        setSystemRole(systemRole);
        return chat(prompt);
    }

    public String getSystemRole() {
        return systemRole;
    }
}