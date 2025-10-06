from src.llm.llm_component import LLMComponent
from src.prompts.detection import DETECTION_PROMPT

class DetectPipeline:
    def __init__(self, model_name: str):
        self.llm = LLMComponent(model_name)

    def detect(self, context: str, prompt: str, response: str) -> str:
        prompt_text = DETECTION_PROMPT.format(
            context=context,
            prompt=prompt,
            response=response
        )
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý phân tích 'ảo giác'."},
            {"role": "user", "content": prompt_text}
        ]
        return self.llm.chat(messages)
