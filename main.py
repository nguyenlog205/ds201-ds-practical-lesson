from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from src.core.logger import setup_logger
from src.utils import (
    load_csv,
    transform_raw_sample_to_input_state,
    save_csv
)
#Test
from src.pipeline.pipeline1 import Pipeline1

logger = setup_logger("DSC_2025_Extrabugs")

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", 
        choices=["warm_up", "train", "public_test", "private_test"], 
        default="warm_up", 
        help="Choose whether to run in warm_up, train, public_test, or private_test mode"
    )
    args = parser.parse_args()

    #File paths
    if args.mode == "warm_up":
        data_file  = Path("./data/warm_up/vihallu-warmup.csv")
    logger.info(f"Data file: {data_file }")

    # Load data
    data = load_csv(data_file)
    submission_file = Path("./submit.csv")
    predictions=[]

    model_name = "Qwen/Qwen2.5-7B-Instruct"  # hoặc lấy từ config
    pipeline = Pipeline1(model_name)
    for idx, row in tqdm(data.iterrows()):
        try:
            logger.info(f"[INFO] Processing Sample {row['id']}")
            input_state = transform_raw_sample_to_input_state(row.to_dict())
            # Dự đoán label bằng pipeline
            predict_label = pipeline.predict(input_state)
            predictions.append({
                "id": input_state.id,
                "predict_label": predict_label.label
            })
            save_csv(submission_file, predictions)
        except Exception as e:
            logger.error(f"[ERROR] Failed to process Sample {row['id']}: {e}")

if __name__ == "__main__":
    main()