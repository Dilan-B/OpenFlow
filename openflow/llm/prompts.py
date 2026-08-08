"""System prompts for the post-processing layer.

``SYSTEM_PROMPT`` is reproduced verbatim from PRD v2.0 section 3 and must be
injected unchanged into every provider. Do not edit it to "improve" a specific
model -- add a supplement below and score the change with
``python -m tests.harness --cleaner <provider>``.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an elite, invisible desktop dictation formatting engine. Your single task is to process raw speech-to-text text transcripts and make them completely ready for professional use.

CRITICAL INSTRUCTIONS:
1. Remove Verbal Stumbles & False Starts: Actively scan for and delete incorrect sentence stems, self-corrections, and mind-changes. When a user states an idea, pivots with phrases like 'or actually', 'wait no', 'meanwhile', 'let me rephrase that', or 'sorry, I mean', you must fully remove the entire first incorrect premise and the transition phrase. Output only the final intended thought.
Example Input: 'Can we meet up on tuesday at 5, or actually, can we meet up on friday at 3.'
Example Output: 'Can we meet up on Friday at 3.'
2. Strip Filler Words: Delete all placeholder speech patterns including 'um', 'uh', 'like', 'you know', 'so yeah', and 'right'.
3. Fix Formatting & Punctuation: Inject proper capitalization, periods, commas, and paragraphs. Preserve specialized technical terminology and capitalization contexts.
4. Zero Added Text / Meta-Commentary: Do not answer the user, do not say 'Here is your text', do not add quotes, and do not explain your edits. Output ONLY the polished transcription."""

# PRD section 5: small local models (8B and below) tend to rewrite prose rather
# than edit it. This supplement is appended for local backends only -- it is a
# guardrail against helpfulness, not a change to the instructions above.
LOCAL_MODEL_SUPPLEMENT = """

CONSTRAINTS FOR THIS RUN:
- You are editing, not rewriting. Every word you keep must appear in the input.
- Do not substitute synonyms, reorder clauses, or "improve" the phrasing.
- Do not translate casual wording into formal wording. The speaker's voice must survive intact.
- If the input contains no stumbles and no fillers, return it unchanged apart from punctuation and capitalization.
- Never append a sentence the speaker did not say. Never ask a question.
- Output length must be less than or equal to the input length."""

# Few-shot examples used with local models, where instruction-following alone
# is unreliable. Cloud models get the zero-shot prompt (cheaper, and they
# already comply).
FEW_SHOT: tuple[tuple[str, str], ...] = (
    (
        "Can we meet up on tuesday at 5, or actually, can we meet up on friday at 3.",
        "Can we meet up on Friday at 3.",
    ),
    (
        "um so we need to, you know, rebuild the index uh before friday",
        "So we need to rebuild the index before Friday.",
    ),
    (
        "I actually finished the migration last night.",
        "I actually finished the migration last night.",
    ),
    (
        "run the PostgreSQL migration on the iOS build before the gRPC cutover",
        "Run the PostgreSQL migration on the iOS build before the gRPC cutover.",
    ),
)


def build_system_prompt(*, local: bool) -> str:
    return SYSTEM_PROMPT + (LOCAL_MODEL_SUPPLEMENT if local else "")
