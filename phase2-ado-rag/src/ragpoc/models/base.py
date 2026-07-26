"""
Abstract base classes for LLM and Embedding providers.

This is the swappability layer — the entire project's portability depends
on these interfaces. Nothing outside models/ should ever import provider
SDKs (ollama, sentence_transformers, etc.) directly.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector for the query.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def generate_with_context(
        self, query: str, context: str, system_prompt: str = ""
    ) -> str:
        """Generate a grounded response given query + retrieved context.

        Args:
            query: The user's question.
            context: Retrieved context passages to ground the answer.
            system_prompt: Optional system instruction.

        Returns:
            Generated text response grounded in the provided context.
        """
        ...

    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical document that *would* answer the query.

        Used by HyDE (Hypothetical Document Embeddings) to close the
        wording gap between user stories and test cases. The returned
        text is embedded for retrieval — it is never shown to the user.

        The default implementation calls generate() with a HyDE-specific
        prompt. Subclasses may override for provider-specific tuning.

        Args:
            query: The user's question or user story text.

        Returns:
            A hypothetical test case / answer document.
        """
        hyde_prompt = (
            "You are a QA engineer. Given the following user story or requirement, "
            "write a detailed test case that would verify this requirement. "
            "Include: Test Case Title, Preconditions, Test Steps (numbered), "
            "and Expected Results. Do NOT add any commentary — output only the "
            "test case.\n\n"
            f"User Story / Requirement:\n{query}"
        )
        return self.generate(hyde_prompt)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and ready.

        Returns:
            True if the provider can accept requests.
        """
        ...

