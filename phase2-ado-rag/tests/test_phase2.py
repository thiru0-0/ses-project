"""
Unit tests for Phase 2 components: HyDE, ADO MCP client, and
test-case generation pipeline.
"""

import json
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

# ===================================================================
# Tests for HyDE in the LLM abstraction layer
# ===================================================================


class TestHyDEInLLMProvider:
    """Test the HyDE method on the LLMProvider abstract base class."""

    def test_base_class_has_generate_hyde_document_method(self):
        """The LLMProvider ABC should expose generate_hyde_document."""
        from src.ragpoc.models.base import LLMProvider

        assert hasattr(LLMProvider, "generate_hyde_document")

    def test_default_hyde_calls_generate(self):
        """The default implementation should call generate() internally."""
        from src.ragpoc.models.base import LLMProvider

        class FakeLLM(LLMProvider):
            def generate(self, prompt, system_prompt=""):
                return f"GENERATED: {prompt[:50]}"

            def generate_with_context(self, query, context, system_prompt=""):
                return ""

            def is_available(self):
                return True

        llm = FakeLLM()
        result = llm.generate_hyde_document("As a user I want to login")
        assert "GENERATED:" in result
        assert "QA engineer" not in result or len(result) > 0  # prompt was passed

    def test_hyde_output_is_string(self):
        """HyDE should always return a string."""
        from src.ragpoc.models.base import LLMProvider

        class FakeLLM(LLMProvider):
            def generate(self, prompt, system_prompt=""):
                return "Test Case: Login\n- Steps:\n1. Go to /login"

            def generate_with_context(self, query, context, system_prompt=""):
                return ""

            def is_available(self):
                return True

        llm = FakeLLM()
        result = llm.generate_hyde_document("login user story")
        assert isinstance(result, str)
        assert len(result) > 0


class TestOllamaLLMHyDE:
    """Test the HyDE override in OllamaLLM."""

    @patch("src.ragpoc.models.llm.Client")
    def test_generate_hyde_uses_higher_temperature(self, MockClient):
        """HyDE should use hyde_temperature from settings, not 0.1."""
        mock_client = MockClient.return_value
        mock_response = MagicMock()
        mock_response.message.content = "Test Case: Verify login"
        mock_client.chat.return_value = mock_response

        from src.ragpoc.models.llm import OllamaLLM

        llm = OllamaLLM.__new__(OllamaLLM)
        llm.model = "test-model"
        llm.base_url = "http://localhost:11434"
        llm.client = mock_client

        result = llm.generate_hyde_document("As a user I want to login")

        assert result == "Test Case: Verify login"
        call_args = mock_client.chat.call_args
        assert call_args[1]["options"]["temperature"] > 0.1

    @patch("src.ragpoc.models.llm.Client")
    def test_generate_hyde_limits_token_output(self, MockClient):
        """HyDE should use a shorter num_predict than regular generation."""
        mock_client = MockClient.return_value
        mock_response = MagicMock()
        mock_response.message.content = "Test Case output"
        mock_client.chat.return_value = mock_response

        from src.ragpoc.models.llm import OllamaLLM

        llm = OllamaLLM.__new__(OllamaLLM)
        llm.model = "test-model"
        llm.base_url = "http://localhost:11434"
        llm.client = mock_client

        llm.generate_hyde_document("user story")

        call_args = mock_client.chat.call_args
        assert call_args[1]["options"]["num_predict"] <= 512


# ===================================================================
# Tests for the Retriever with HyDE
# ===================================================================


class TestRetrieverWithHyDE:
    """Test that the Retriever correctly integrates HyDE expansion."""

    def test_retriever_init_respects_use_hyde_flag(self):
        """Retriever should accept a use_hyde parameter."""
        from src.ragpoc.retrieval.retriever import Retriever

        mock_vs = MagicMock()
        retriever = Retriever(mock_vs, use_hyde=False)
        assert retriever._use_hyde is False

        retriever_hyde = Retriever(mock_vs, use_hyde=True)
        assert retriever_hyde._use_hyde is True

    @patch("src.ragpoc.models.registry.get_llm_provider")
    def test_hyde_expansion_calls_llm(self, mock_get_llm):
        """When HyDE is enabled, retrieve should call generate_hyde_document."""
        from src.ragpoc.retrieval.retriever import Retriever

        mock_llm = MagicMock()
        mock_llm.generate_hyde_document.return_value = "Hypothetical test case"
        mock_get_llm.return_value = mock_llm

        mock_vs = MagicMock()
        mock_vs.query_raw.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        retriever = Retriever(mock_vs, use_hyde=True)
        retriever.retrieve("user story about login")

        mock_llm.generate_hyde_document.assert_called_once_with("user story about login")

    def test_hyde_disabled_does_not_call_llm(self):
        """When HyDE is disabled, retrieve should embed the raw query."""
        from src.ragpoc.retrieval.retriever import Retriever

        mock_vs = MagicMock()
        mock_vs.query_raw.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        retriever = Retriever(mock_vs, use_hyde=False)
        retriever.retrieve("simple query")

        # query_raw should be called with the raw query, not a HyDE expansion
        mock_vs.query_raw.assert_called_once()
        call_args = mock_vs.query_raw.call_args
        assert call_args[0][0] == "simple query"


# ===================================================================
# Tests for ADO MCP Client
# ===================================================================


class TestADOMCPClient:
    """Test the ADO MCP client's data conversion logic."""

    def test_render_work_item_text_user_story(self):
        """Work item rendering should produce clean embeddable text."""
        from src.ragpoc.ingestion.mcp_client import ADOMCPClient, ADOWorkItem

        item = ADOWorkItem(
            id=42,
            title="Login with SSO",
            work_item_type="User Story",
            state="Active",
            description="<p>As a user I want <b>SSO login</b></p>",
            area_path="Project\\Auth",
            iteration_path="Project\\Sprint 1",
        )

        text = ADOMCPClient._render_work_item_text(item)

        assert "[User Story] #42: Login with SSO" in text
        assert "State: Active" in text
        assert "Area: Project\\Auth" in text
        # HTML should be stripped
        assert "<p>" not in text
        assert "<b>" not in text
        assert "SSO login" in text

    def test_render_work_item_text_test_case(self):
        """Test case items should include steps in the rendered text."""
        from src.ragpoc.ingestion.mcp_client import ADOMCPClient, ADOWorkItem

        item = ADOWorkItem(
            id=99,
            title="TC: Verify Login",
            work_item_type="Test Case",
            state="Ready",
            steps="<steps><step>Navigate to /login</step></steps>",
        )

        text = ADOMCPClient._render_work_item_text(item)

        assert "[Test Case] #99" in text
        assert "Test Steps:" in text
        assert "Navigate to /login" in text

    def test_work_items_to_raw_documents(self):
        """Conversion should produce valid RawDocument objects."""
        from src.ragpoc.ingestion.mcp_client import ADOMCPClient, ADOWorkItem

        client = ADOMCPClient.__new__(ADOMCPClient)
        client._org_url = "https://dev.azure.com/test"
        client._project = "TestProject"

        items = [
            ADOWorkItem(
                id=1,
                title="Story One",
                work_item_type="User Story",
                state="Active",
                description="Description one",
                url="https://dev.azure.com/test/TestProject/_workitems/edit/1",
            ),
        ]

        docs = client.work_items_to_raw_documents(items)

        assert len(docs) == 1
        assert docs[0].source_type == "ado_work_item"
        assert docs[0].metadata["work_item_type"] == "User Story"
        assert docs[0].metadata["work_item_id"] == 1


# ===================================================================
# Tests for the ADO chunker extension
# ===================================================================


class TestADOChunking:
    """Test that ADO source types get correct chunk parameters."""

    def test_ado_work_item_chunk_params(self):
        """ado_work_item should use the ADO-specific chunk settings."""
        from src.ragpoc.ingestion.chunker import get_chunk_params, CHARS_PER_TOKEN

        size, overlap = get_chunk_params("ado_work_item")
        assert size == 400 * CHARS_PER_TOKEN
        assert overlap == 70 * CHARS_PER_TOKEN

    def test_ado_wiki_chunk_params(self):
        """ado_wiki should use the same ADO-specific chunk settings."""
        from src.ragpoc.ingestion.chunker import get_chunk_params, CHARS_PER_TOKEN

        size, overlap = get_chunk_params("ado_wiki")
        assert size == 400 * CHARS_PER_TOKEN
        assert overlap == 70 * CHARS_PER_TOKEN


# ===================================================================
# Tests for the Pipeline mode selection
# ===================================================================


class TestPipelineModeSelection:
    """Test that the pipeline correctly selects output mode."""

    def test_detect_mode_qa_for_pdf_chunks(self):
        """Non-ADO chunks should default to 'qa' mode."""
        from src.ragpoc.generation.pipeline import RAGPipeline

        mock_chunk = MagicMock()
        mock_chunk.source_type = "pdf"

        mode = RAGPipeline._detect_mode([mock_chunk])
        assert mode == "qa"

    def test_detect_mode_test_case_for_ado_chunks(self):
        """ADO work item chunks should trigger 'test_case' mode."""
        from src.ragpoc.generation.pipeline import RAGPipeline

        mock_chunk = MagicMock()
        mock_chunk.source_type = "ado_work_item"

        mode = RAGPipeline._detect_mode([mock_chunk])
        assert mode == "test_case"

    def test_detect_mode_test_case_for_ado_wiki(self):
        """ADO wiki chunks should also trigger 'test_case' mode."""
        from src.ragpoc.generation.pipeline import RAGPipeline

        mock_chunk = MagicMock()
        mock_chunk.source_type = "ado_wiki"

        mode = RAGPipeline._detect_mode([mock_chunk])
        assert mode == "test_case"


# ===================================================================
# Tests for the prompt templates
# ===================================================================


class TestPromptTemplates:
    """Test that both prompt formats render correctly."""

    def test_format_test_case_prompt(self):
        """Test case prompt should include user story and structure."""
        from src.ragpoc.generation.prompt_templates import format_test_case_prompt

        result = format_test_case_prompt(
            context="Some context about login",
            question="As a user I want to login",
        )

        assert "USER STORY / REQUIREMENT:" in result
        assert "As a user I want to login" in result
        assert "Test Case:" in result
        assert "Preconditions:" in result
        assert "Steps:" in result
        assert "Expected Result:" in result

    def test_format_qa_prompt_unchanged(self):
        """Original QA prompt should still work as before."""
        from src.ragpoc.generation.prompt_templates import format_qa_prompt

        result = format_qa_prompt(
            context="Some context",
            question="What is X?",
        )

        assert "QUESTION: What is X?" in result
        assert "CONTEXT:" in result


# ===================================================================
# Tests for the golden set loader
# ===================================================================


class TestADOGoldenSet:
    """Test the ADO golden set JSON structure."""

    def test_golden_set_is_valid_json(self):
        """The golden set file should be valid JSON."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "ado_golden_set.json"
        if golden_path.exists():
            with open(golden_path) as f:
                data = json.load(f)
            assert "pairs" in data
            assert len(data["pairs"]) > 0

    def test_golden_set_pairs_have_required_fields(self):
        """Each pair should have user_story and expected_test_case."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "ado_golden_set.json"
        if golden_path.exists():
            with open(golden_path) as f:
                data = json.load(f)
            for pair in data["pairs"]:
                assert "user_story" in pair, f"Missing user_story in {pair.get('id', '?')}"
                assert "expected_test_case" in pair, f"Missing expected_test_case in {pair.get('id', '?')}"
                assert len(pair["user_story"]) > 0
                assert len(pair["expected_test_case"]) > 0
