"""Prompt templates for the sales call AI copilot."""

SYSTEM_PROMPT = """\
You are a sales call analysis assistant. You help sales teams understand and \
extract insights from their recorded call transcripts.

Rules:
- Always cite the source transcript segments you used in your answer.
- Use the citation format: [Call #<call_id>, <start_time>-<end_time>]
- If multiple segments are relevant, cite each one.
- If the provided context does not contain enough information to answer, say so clearly.
- Be concise, specific, and actionable in your responses."""

QA_PROMPT = """\
Based on the following transcript segments, answer the user's question.
Cite which segments you used with the format [Call #<call_id>, <start_time>-<end_time>].

--- TRANSCRIPT SEGMENTS ---
{context}
--- END SEGMENTS ---

Question: {query}"""

SUMMARY_PROMPT = """\
Summarize the following call transcript segments. Include:
- Key topics discussed
- Decisions made
- Action items with owners
- Notable quotes with timestamps

Cite sources with [Call #<call_id>, <start_time>-<end_time>].

--- TRANSCRIPT SEGMENTS ---
{context}
--- END SEGMENTS ---"""

SENTIMENT_PROMPT = """\
Analyze the sentiment in these transcript segments. For each notable statement:
- Classify as positive, negative, or neutral
- Attribute to the speaker
- Provide the timestamp and quote

Cite sources with [Call #<call_id>, <start_time>-<end_time>].

--- TRANSCRIPT SEGMENTS ---
{context}
--- END SEGMENTS ---

Focus on: {query}"""
