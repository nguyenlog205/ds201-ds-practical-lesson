
from utils import Vocabulary, TextDataset
from utils import preprocess_dataset, collate_fn
from model.lstm import lstm

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split

BASE_DIR = r'self-01/'
DATA_DIR = BASE_DIR + 'data/'

# Vocab configuration
FREQ_THRESHOLD = 2

# Dataloader Configuration
BATCH_SIZE = 32
TRAIN_DEV_RATE = 0.15

# Model configuration
D_MODEL = 512
N_CLASSES = 2
DROPOUT_RATE = 0.3
N_LAYERS=8

# Training
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
PAD_IDX = 2
BEST_MODEL_PATH = BASE_DIR + r'checkpoints/best_model.pth'
HISTORY_PATH = BASE_DIR + r'checkpoints/history.json'
PATIENCE = 10

# PREIDTC
OUTPUT_PATH = DATA_DIR + r'submission.csv'

def train():
    # Tải và tiền xử lý bộ dữ liệu
    raw_train_df = pd.read_csv(DATA_DIR + 'train.csv')
    raw_test_df = pd.read_csv(DATA_DIR + 'test.csv')
    train_df, dev_df = train_test_split(raw_train_df, test_size=TRAIN_DEV_RATE, random_state=42)

    train_df = preprocess_dataset(train_df.reset_index(drop=True) )
    dev_df = preprocess_dataset(dev_df.reset_index(drop=True))
    test_df = preprocess_dataset(raw_test_df.reset_index(drop=True))
    placeholder_labels = [1] * len(test_df)

    # Xây dựng bộ từ vựng
    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build_vocabulary(
        train_df['tokens'].tolist()
    )
    print(f'Xây dựng xong từ vựng, tổng cộng có {vocab.n_words} từ!')

    # Xây dựng dataset
    train_dataset = TextDataset(train_df['tokens'].tolist() ,train_df['target'].tolist()  ,vocab=vocab)
    dev_dataset   = TextDataset(dev_df['tokens'].tolist()   ,dev_df['target'].tolist()    ,vocab=vocab)
    test_dataset  = TextDataset(test_df['tokens'].tolist()  ,placeholder_labels           ,vocab=vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, PAD_ID=PAD_IDX)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, PAD_ID=PAD_IDX)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, PAD_ID=PAD_IDX)
    )
    print(f'Xây dựng xong bộ dữ liệu và các loader!')

    # Khởi tạo model và tham số train 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = lstm(
        vocab_size=vocab.n_words,
        d_model = D_MODEL,
        num_classes=N_CLASSES,
        dropout=DROPOUT_RATE,
        num_layers=N_LAYERS,
        pad_idx=PAD_IDX 
    )
    model.to(device)
    class_weights = torch.tensor(
        [
            len(train_df) / train_df['target'].value_counts()[0], 
            len(train_df) / train_df['target'].value_counts()[1]
        ], 
        dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f'Khởi tạo xong mô hình {model}')

    history = {
        'train_loss': [],
        'dev_loss': [],
        'dev_accuracy': []
    }
    patience = 10
    min_delta = 1e-5
    best_dev_loss = float('inf')
    epochs_no_improve = 0

    print(f'Bắt đầu training!')
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for text, labels in train_loader:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_dev_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for text, labels in dev_loader:
                text, labels = text.to(device), labels.to(device)

                outputs = model(text)
                loss = criterion(outputs, labels)
                total_dev_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_dev_loss = total_dev_loss / len(dev_loader)
        dev_accuracy = correct_predictions / total_samples


        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Dev Acc: {dev_accuracy:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_accuracy'].append(dev_accuracy)

        if avg_dev_loss < best_dev_loss - min_delta:
            print(f"Dev Loss cải thiện ({best_dev_loss:.4f} -> {avg_dev_loss:.4f}). Lưu mô hình...")
            best_dev_loss = avg_dev_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1
            print(f"Dev Loss không cải thiện. Chờ: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("--- Kích hoạt Early Stopping! ---")
            break
    

    import json
    import os
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
    print(f"Lịch sử huấn luyện đã được lưu tại {HISTORY_PATH}")
    print(f"Mô hình tốt nhất đã được lưu tại {BEST_MODEL_PATH}")





    
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Tải lại trọng số tốt nhất từ {BEST_MODEL_PATH}")
        # Tải trọng số đã lưu (model.state_dict())
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("CẢNH BÁO: Không tìm thấy BEST_MODEL_PATH. Sử dụng mô hình từ epoch cuối cùng.")

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for text_tensor, _ in test_loader:
            text_tensor = text_tensor.to(device)
            outputs = model(text_tensor)
            _, predicted_classes = torch.max(outputs, 1)
            all_predictions.extend(predicted_classes.cpu().tolist())

    print(f"Tổng số dự đoán: {len(all_predictions)}")
    print(f"Tổng số mẫu test: {len(test_df)}")

    submission_df = pd.DataFrame({
        'id': test_df['id'].tolist(),
        'target': all_predictions
    })
    
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Kết quả dự đoán đã được lưu tại: {OUTPUT_PATH}")

train()