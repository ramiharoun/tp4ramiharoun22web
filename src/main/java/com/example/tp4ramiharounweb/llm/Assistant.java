package com.example.tp4ramiharounweb.llm;



/**
 * Interface "service IA" décrivant la communication avec le LLM.
 * LangChain4j va automatiquement générer une implémentation concrète
 * à partir de cette interface.
 */
public interface Assistant {

    /**
     * Méthode principale pour dialoguer avec le LLM.
     * @param prompt question ou message envoyé au modèle
     * @return réponse générée par le LLM
     */
    String chat(String prompt);
}
