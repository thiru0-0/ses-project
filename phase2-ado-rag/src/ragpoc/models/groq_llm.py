"""
LLM provider using Groq cloud API with Gemma-2 9B.

Drop-in replacement for OllamaLLM — same interface, cloud-hosted inference.
"""

import logging
import time

from groq import Groq

from src.ragpoc.models.base import LLMProvider
from config.settings import settings

logger = logging.getLogger(__name__)


class GroqLLM(LLMProvider):
    """Cloud LLM provider using Groq."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.groq_model
        self.client = Groq(api_key=api_key or settings.groq_api_key)
        logger.info("Initialized Groq LLM: model='%s'", self.model)

    def _chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Send a chat completion request to Groq."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._chat(messages)

    def generate_with_context(
        self, query: str, context: str, system_prompt: str = ""
    ) -> str:
        """Generate a grounded response given query + retrieved context."""
        full_prompt = (
            f"CONTEXT:\n---\n{context}\n---\n\n"
            f"QUESTION: {query}\n\n"
            "Answer the question using ONLY the context above. "
            "Cite sources using [Source: <title>]."
        )
        return self.generate(full_prompt, system_prompt=system_prompt)

    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical test case for HyDE retrieval."""
        hyde_prompt = (
            "You are a QA engineer. Given the following user story or requirement, "
            "write a detailed test case that would verify this requirement. "
            "Include: Test Case Title, Preconditions, Test Steps (numbered), "
            "and Expected Results. Do NOT add any commentary — output only the "
            "test case.\n\n"
            f"User Story / Requirement:\n{query}"
        )

        messages = [{"role": "user", "content": hyde_prompt}]

        t0 = time.perf_counter()
        result = self._chat(messages, temperature=settings.hyde_temperature, max_tokens=512)
        elapsed = time.perf_counter() - t0
        logger.info(
            "HyDE document generated in %.2fs (%d chars)", elapsed, len(result)
        )
        return result

    def is_available(self) -> bool:
        """Check if Groq API is reachable and the key is valid."""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error("Groq health check failed: %s", e)
            return False
