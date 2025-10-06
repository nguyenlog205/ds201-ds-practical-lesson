from src.llm.llm_component import LLMComponent
from src.prompts.predict import PREDICT_PROMPT

class PredictPipeline:
    def __init__(self, model_name: str):
        self.llm = LLMComponent(model_name)

    def predict(self, context: str, prompt: str, response: str,hallucinations:str) -> dict:
        prompt_text = PREDICT_PROMPT.format(
            context=context,
            prompt=prompt,
            response=response,
            hallucinations=hallucinations
        )
        messages = [{"role": "user", "content": prompt_text}]
        return self.llm.chat(messages)