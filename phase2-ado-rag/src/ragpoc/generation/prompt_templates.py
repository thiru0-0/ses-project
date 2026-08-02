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
    "5. Generate a unique Test ID for each test case (e.g., TC_LOGIN_01).\n"
    "6. Include Requirement Reference from the context if available.\n"
    "7. Format output as JSON array of test cases."
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

# Test-case-shaped output (Phase 2) - Detailed format
TEST_CASE_PROMPT_TEMPLATE = (
    "CONTEXT:\n"
    "---\n"
    "{context}\n"
    "---\n"
    "\n"
    "USER STORY / REQUIREMENT:\n"
    "{question}\n"
    "\n"
    "Generate one or more detailed test cases that verify the above requirement. "
    "Use ONLY the context for facts. Each test case should clearly specify:\n"
    "  Test Case: <unique test id> - <title>\n"
    "  Preconditions: <what must be true before testing>\n"
    "  Steps: <the numbered steps to perform>\n"
    "  Expected Result: <the expected outcome>\n\n"
    "Format each test case as a JSON object with the following fields:\n\n"
    "{{\n"
    '  "test_id": "TC_LOGIN_01",\n'
    '  "requirement_reference": "REQ-001",\n'
    '  "title": "Verify successful login with correct credentials",\n'
    '  "description": "A short sentence explaining what is being tested",\n'
    '  "preconditions": ["User has an active account", "User is on the login screen"],\n'
    '  "test_steps": [\n'
    '    "Enter user@test.com in the email field",\n'
    '    "Enter Password123 in the password field",\n'
    '    "Click the \\"Sign In\\" button"\n'
    '  ],\n'
    '  "expected_result": "The system logs the user in and opens the home dashboard",\n'
    '  "actual_result": "",\n'
    '  "status": "Not Run"\n'
    "}}\n\n"
    "Cite sources using [Source: <title>] within the description or steps if applicable.\n"
    "Return a JSON array of test cases. If multiple test cases, increment the Test ID (e.g., TC_LOGIN_01, TC_LOGIN_02)."
)

# --- Decline Response ---
# Static string returned when the relevance grader declines.
# No LLM call is made — this is returned directly.
DECLINE_RESPONSE = (
    "I couldn't find this information in the provided sources. "
    "The documents in the current workspace do not contain relevant "
    "information to answer your question."
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
