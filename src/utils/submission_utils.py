from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from src.models.global_state import GlobalInputState
import csv

def transform_raw_sample_to_input_state(
    sample: Dict[str, Any]
) -> GlobalInputState:
    return GlobalInputState(
        id=sample["id"],
        context=sample["context"],
        prompt=sample["prompt"],
        response=sample["response"]
    )

def save_csv(pred_file: Union[str, Path], results: List[Dict[str, Any]]):
    if not results:
        return
    with open(pred_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)