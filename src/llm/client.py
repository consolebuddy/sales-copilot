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
        history: list[dict] | None = None,
    ) -> str:
        """Call the chat completion API and return the response text.

        If *history* is provided, it is injected between the system prompt and
        the current user message so the LLM can resolve follow-up references
        like "tell me more" or "what about the second point?".
        """
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_prompt})

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            return "Error: OpenAI rate limit reached. Please wait and try again."
        except APIError as e:
            return f"Error: OpenAI API error — {e.message}"

    def route_query(self, user_input: str, history: list[dict] | None = None) -> list[dict]:
        """Use function calling to classify intent and extract parameters.

        Returns a list of action dicts. A single message may produce multiple
        actions (e.g. "ingest call 5 and delete call 3" -> two actions).

        Each dict has: action, call_ids, file_paths, query.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "route_intent",
                    "description": (
                        "Route ONE user intent. If the user's message contains "
                        "multiple distinct intents (e.g. 'ingest X and also delete Y'), "
                        "call this function once per intent."
                    ),
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
                                "description": "List of numeric call IDs for this action (e.g. 'call 1 and call 6' -> ['1', '6']). Empty array if no specific call.",
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
            "Analyze the user's message and call the route_intent function. "
            "If the message contains MULTIPLE distinct intents (e.g. 'add X and delete Y'), "
            "call route_intent ONCE PER INTENT — do not merge them into one call. "
            "Be precise about extracting call IDs and file paths. "
            "Use the conversation history (if any) to resolve references like "
            "'tell me more', 'that call', 'the same one', etc."
        )

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                temperature=0,
            )

            actions = []
            for tool_call in response.choices[0].message.tool_calls:
                actions.append(json.loads(tool_call.function.arguments))
            return actions if actions else [{"action": "question", "query": user_input}]

        except (RateLimitError, APIError, Exception):
            return [{"action": "question", "query": user_input}]
