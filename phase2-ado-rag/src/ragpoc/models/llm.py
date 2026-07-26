"""
LLM provider using Ollama running locally with Qwen2.5:14B-Instruct.

Fully local — no network calls outside localhost.
"""

import logging
import time

from ollama import Client, ResponseError

from src.ragpoc.models.base import LLMProvider
from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """Local LLM provider using Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = Client(host=self.base_url)
        logger.info(
            "Initialized Ollama LLM: model='%s', base_url='%s'",
            self.model,
            self.base_url,
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,  # low temp for factual grounding
                    "num_predict": 1024,
                },
            )
            # Support both typed response objects and plain dicts
            msg = response.message if hasattr(response, "message") else response["message"]
            return msg.content if hasattr(msg, "content") else msg["content"]
        except ResponseError as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def generate_with_context(
        self, query: str, context: str, system_prompt: str = ""
    ) -> str:
        """Generate a grounded response given query + retrieved context.

        Builds a full prompt with context and query, then calls generate().
        """
        full_prompt = (
            f"CONTEXT:\n---\n{context}\n---\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer the question using ONLY the context above. "
            f"Cite sources using [Source: <title>]."
        )
        return self.generate(full_prompt, system_prompt=system_prompt)

    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical test case for HyDE retrieval.

        Uses a slightly higher temperature than factual generation
        to produce a plausible-but-creative hypothetical answer.
        The output is embedded for vector similarity — never shown to users.
        """
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
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": settings.hyde_temperature,
                    "num_predict": 512,  # shorter — only need enough for embedding
                },
            )
            msg = response.message if hasattr(response, "message") else response["message"]
            result = msg.content if hasattr(msg, "content") else msg["content"]
            elapsed = time.perf_counter() - t0
            logger.info(
                "HyDE document generated in %.2fs (%d chars)",
                elapsed,
                len(result),
            )
            return result
        except ResponseError as e:
            logger.error("HyDE generation failed: %s", e)
            raise

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            # List available models to verify connectivity
            # Newer Ollama client returns a typed ListResponse with .models
            response = self.client.list()
            model_list = getattr(response, "models", None) or response.get("models", [])
            model_names = [
                getattr(m, "model", None) or m.get("name", "") or m.get("model", "")
                for m in model_list
            ]
            # Check if our target model (or a variant) is available
            available = any(
                self.model in name or name.startswith(self.model)
                for name in model_names
            )
            if not available:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self.model,
                    model_names,
                )
            return available
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False


def get_ollama_llm() -> OllamaLLM:
    """Create and return an OllamaLLM instance."""
    return OllamaLLM()

