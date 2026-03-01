"""OpenAI LLM client wrapper."""

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
