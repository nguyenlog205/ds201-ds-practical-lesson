import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from config import Config
from dataset import TextDataset, collate_fn
from model import DisasterClassifier
from utils import load_vocab, load_config # Import hàm load mới

def run_inference():
    # 1. Setup
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. LOAD ARTIFACTS (Thay vì build lại)
    print(f"Loading artifacts from {Config.CHECKPOINT_DIR}...")
    
    # a. Load Vocab
    if not os.path.exists(Config.VOCAB_PATH):
        raise FileNotFoundError(f"Không tìm thấy {Config.VOCAB_PATH}. Hãy train trước!")
    vocab = load_vocab(Config.VOCAB_PATH)
    print(f"✅ Vocab loaded (Size: {len(vocab)})")
    
    # b. Load Config (để lấy tham số model cũ)
    saved_config = load_config(Config.CONFIG_PATH)
    pad_idx = vocab["<PAD>"]

    # 3. Load Test Data
    print("Loading Test Data...")
    test_df = pd.read_csv(Config.TEST_PATH)
    
    # Tạo Dataset (Dùng vocab vừa load)
    test_dataset = TextDataset(test_df, vocab, is_test=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    # 4. Init Model (Dùng tham số từ saved_config để đảm bảo khớp 100%)
    print("Initializing Model...")
    model = DisasterClassifier(
        vocab_size=len(vocab),
        d_model=saved_config['D_MODEL'],
        num_classes=saved_config['N_CLASSES'],
        dropout=saved_config['DROPOUT'],
        num_layers=saved_config['N_LAYERS'],
        pad_idx=pad_idx
    ).to(device)

    # 5. Load Weights
    print(f"Loading weights from {Config.MODEL_PATH}...")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 6. Prediction (Giữ nguyên)
    print("Running Prediction...")
    all_ids = []
    all_predictions = []

    with torch.no_grad():
        for inputs, ids in test_loader: # Nhớ: dataset test trả về (inputs, ids)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_ids.extend(ids.tolist())
            all_predictions.extend(predicted.cpu().tolist())

    # 7. Save
    submission_df = pd.DataFrame({'id': all_ids, 'target': all_predictions})
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"✅ Done! Submission saved to: {Config.SUBMISSION_PATH}")

if __name__ == "__main__":
    run_inference()