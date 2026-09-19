import requests

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    DEFAULT_TEMPERATURE,
)


class LLMClient:

    def __init__(self):
        self.url = f"{LM_STUDIO_BASE_URL}/chat/completions"
        self.model = LM_STUDIO_MODEL

    def generate(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
        }

        response = requests.post(
            self.url,
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]