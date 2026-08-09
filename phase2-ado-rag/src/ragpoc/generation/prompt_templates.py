"""
Prompt templates for the RAG pipeline.

The system prompt is STRICT — this has a bigger impact on hallucination
reduction than changing between similarly capable models.

The decline response is a static string, NOT an LLM generation. When
the relevance grader determines no chunk is relevant enough, the
pipeline short-circuits and returns this directly. This guarantees
zero hallucination on out-of-scope questions.

Phase 2 adds a test-case-shaped output format for ADO queries, alongside
the original free-text QA format for general document queries.
"""

# --- System Prompts ---
# Strict retrieval assistant prompt — general documents (Phase 1)
SYSTEM_PROMPT = (
    "You are a retrieval assistant. Answer only using the provided context.\n"
    "If the answer is not present in the context, reply:\n"
    '"I couldn\'t find this information in the provided sources."\n'
    "Do not use prior knowledge. Always cite the relevant source(s)."
)

# Test-case generation system prompt — ADO / Phase 2
SYSTEM_PROMPT_TEST_CASE = (
    "You are a QA test case generation assistant. Your job is to generate "
    "structured test cases ONLY from the provided context.\n\n"
    "Rules:\n"
    "1. Use ONLY information present in the context — do NOT invent requirements.\n"
    "2. If the context does not contain enough information to write a meaningful "
    "test case, reply: \"Insufficient context to generate a test case.\"\n"
    "3. Always cite the source work item or document using [Source: <title>].\n"
    "4. Do not use prior knowledge.\n"
    "5. Generate ALL fields in the structured format — leave Actual Result and "
    "Status as placeholders for QA execution."
)

# --- QA Prompt Templates ---
# General free-text QA (Phase 1)
QA_PROMPT_TEMPLATE = (
    "CONTEXT:\n"
    "---\n"
    "{context}\n"
    "---\n"
    "\n"
    "QUESTION: {question}\n"
    "\n"
    "Answer the question using ONLY the context above. "
    "Cite sources using [Source: <title>]."
)

# Test-case-shaped output with full 8-field format (Phase 2)
TEST_CASE_PROMPT_TEMPLATE = (
    "CONTEXT:\n"
    "---\n"
    "{context}\n"
    "---\n"
    "\n"
    "USER STORY / REQUIREMENT:\n"
    "{question}\n"
    "\n"
    "Generate one or more test cases that verify the above requirement. "
    "Use ONLY the context for facts. Format each test case with ALL 8 fields "
    "exactly as shown:\n\n"
    "**Test Case:**\n"
    "| Field | Value |\n"
    "|---|---|\n"
    "| **Test ID** | TC-<number> |\n"
    "| **Requirement Reference** | <source work item ID or document ref> |\n"
    "| **Title / Description** | <concise test case title> |\n"
    "| **Preconditions** | <list preconditions or 'None'> |\n"
    "| **Test Steps** | 1. <step> 2. <step> ... |\n"
    "| **Expected Result** | <what should happen> |\n"
    "| **Actual Result** | _To be filled during execution_ |\n"
    "| **Status** | _Not Executed_ |\n\n"
    "Cite sources using [Source: <title>]."
)

# --- Decline Response ---
# Static string returned when the relevance grader declines.
# No LLM call is made — this is returned directly.
DECLINE_RESPONSE = (
    "I couldn't find this information in the provided sources. "
    "The documents in the current workspace do not contain relevant "
    "information to answer your question."
)

# --- Low Confidence Response ---
# Appended to the answer when confidence is below the human-review threshold.
# The answer IS still generated, but flagged for human review.
LOW_CONFIDENCE_FLAG = (
    "\n\n---\n"
    "⚠️ **Human Review Recommended**: This answer was generated with low "
    "confidence. Please verify the output against the original source "
    "documents before using it."
)


def format_qa_prompt(context: str, question: str) -> str:
    """Format the QA prompt with context and question.

    Args:
        context: Formatted context string from retrieved chunks.
        question: The user's question.

    Returns:
        Formatted prompt string ready for LLM.
    """
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def format_test_case_prompt(context: str, question: str) -> str:
    """Format the test-case generation prompt with context and user story.

    Args:
        context: Formatted context string from retrieved chunks.
        question: The user story or requirement text.

    Returns:
        Formatted prompt string for test-case generation.
    """
    return TEST_CASE_PROMPT_TEMPLATE.format(context=context, question=question)


def format_context_from_chunks(chunks) -> str:
    """Format retrieved chunks into a context string for the prompt.

    Each chunk is labelled with its source title for citation purposes.

    Args:
        chunks: List of RetrievedChunk objects.

    Returns:
        Formatted context string.
    """
    context_parts = []
    seen_titles = set()

    for i, chunk in enumerate(chunks, 1):
        title = getattr(chunk, "title", "Untitled")
        source_ref = getattr(chunk, "source_ref", "unknown")

        # Build source label
        if title not in seen_titles:
            seen_titles.add(title)

        context_parts.append(
            f"[Source: {title}] (from: {source_ref})\n{chunk.content}"
        )

    return "\n\n".join(context_parts)
