from src.pipeline.predict import PredictPipeline
from src.pipeline.detect import DetectPipeline
from src.utils import extract_label_from_llm_response
import ast
from src.models.global_state import GlobalInputState, GlobalOutputState

class Pipeline1:
    def __init__(self, model_name: str):
        self.detect_pipeline = DetectPipeline(model_name)
        self.predict_pipeline = PredictPipeline(model_name)

    def predict(self, input_state: GlobalInputState) -> str:
        # input_state: GlobalInputState (pydantic model)
        context = input_state.context
        prompt = input_state.prompt
        response = input_state.response
        # Detect hallucinations trước
        detected_str = self.detect_pipeline.detect(context, prompt, response)
        print("[DEBUG] detect output:", detected_str)
        detected = ast.literal_eval(detected_str)
        hallucinations = detected["hallucinations"]
        # Predict với hallucinations vừa detect
        result = self.predict_pipeline.predict(context, prompt, response, hallucinations=hallucinations)
        predicted_label = extract_label_from_llm_response(result)
        return GlobalOutputState(label=predicted_label)