"""OpenAI LLM client wrapper."""

import json
from openai import OpenAI, APIError, RateLimitError

import config


class LLMClient:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.LLM_MODEL

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Call the chat completion API and return the response text."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            return "Error: OpenAI rate limit reached. Please wait and try again."
        except APIError as e:
            return f"Error: OpenAI API error — {e.message}"

    def route_query(self, user_input: str) -> dict:
        """Use function calling to classify intent and extract parameters.

        Returns a dict like:
            {"action": "list_calls"}
            {"action": "ingest", "file_path": "./transcripts/call_5.txt"}
            {"action": "delete_call", "call_id": "5"}
            {"action": "summarize", "query": "...", "call_id": "1"}
            {"action": "sentiment", "query": "...", "call_id": null}
            {"action": "question", "query": "...", "call_id": null}
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "route_intent",
                    "description": "Classify the user's intent and extract parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "list_calls",
                                    "ingest",
                                    "delete_call",
                                    "summarize",
                                    "sentiment",
                                    "question",
                                ],
                                "description": (
                                    "list_calls: user wants to see ingested calls or total count. "
                                    "ingest: user wants to add/load a new transcript file. "
                                    "delete_call: user wants to delete/remove a call. "
                                    "summarize: user wants a summary/recap/overview of call(s). "
                                    "sentiment: user wants sentiment analysis, negative/positive comments, objections, or concerns. "
                                    "question: any other question about the call transcripts."
                                ),
                            },
                            "call_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of numeric call IDs the user references (e.g. 'call 1 and call 6' -> ['1', '6'], 'call_3' -> ['3']). Empty array if no specific call.",
                            },
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file paths for ingest action. Extract the exact paths the user provided. Empty array if not applicable.",
                            },
                            "query": {
                                "type": "string",
                                "description": "The user's original query text, to be used for retrieval.",
                            },
                        },
                        "required": ["action", "query"],
                    },
                },
            }
        ]

        system = (
            "You are a query router for a sales call transcript chatbot. "
            "Analyze the user's message and call the route_intent function "
            "with the correct action and parameters. "
            "Be precise about extracting call IDs and file paths."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_input},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "route_intent"}},
                temperature=0,
            )

            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)

        except (RateLimitError, APIError, Exception):
            # Fallback: treat as a general question
            return {"action": "question", "query": user_input}
