import os
from openai import OpenAI

class LLMComponent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", "")
        )

    def chat(self, messages, max_new_tokens=40):
        completion = self.client.chat.completions.create(
            model=f"{self.model_name}:together",
            messages=messages,
        )
        return completion.choices[0].message.content
