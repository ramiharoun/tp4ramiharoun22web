package com.example.tp4ramiharounweb.llm;

import jakarta.enterprise.context.Dependent;
import java.io.Serializable;

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

        // 3. Création de l'assistant via AiServices
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
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
