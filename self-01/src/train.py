import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import Config
from dataset import TextDataset, build_vocab, collate_fn
from model import DisasterClassifier
from utils import calculate_metrics, save_checkpoint, load_glove_embeddings

def main():
    print("Loading Data...")
    raw_train = pd.read_csv(Config.TRAIN_PATH)
    raw_test = pd.read_csv(Config.TEST_PATH)
    
    train_df, dev_df = train_test_split(raw_train, test_size=Config.TRAIN_DEV_SPLIT, random_state=42, stratify=raw_train['target'])

    print("Building Vocabulary...")
    vocab = build_vocab(train_df['text'].tolist(), Config.FREQ_THRESHOLD)
    Config.PAD_IDX = vocab["<PAD>"]
    print(f"Vocab size: {len(vocab)}")
    
    train_ds = TextDataset(train_df, vocab)
    dev_ds = TextDataset(dev_df, vocab)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, Config.PAD_IDX))
    dev_loader = DataLoader(dev_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            collate_fn=lambda b: collate_fn(b, Config.PAD_IDX))
    
    embedding_matrix = load_glove_embeddings(
        vocab=vocab,
        glove_path=Config.GLOVE_PATH,
        d_model=Config.D_MODEL
    )
    # ----------------------------

    print("Initializing Model...")
    model = DisasterClassifier(
        vocab_size=len(vocab),
        d_model=Config.D_MODEL,
        num_classes=Config.N_CLASSES,
        dropout=Config.DROPOUT,
        num_layers=Config.N_LAYERS,
        pad_idx=Config.PAD_IDX,
        embedding_matrix=embedding_matrix
    ).to(Config.DEVICE)
    
    # 4. Loss & Optimizer
    
    class_counts = train_df['target'].value_counts().sort_index()
    weights = torch.tensor(
        [class_counts[1]/class_counts[0], 1.0],
        dtype=torch.float
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights) 
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # 5. Training Loop
    best_f1 = 0.0
    
    print("Start Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
            if X.max() >= len(vocab):
                print(f"🔴 CẢNH BÁO LỖI DỮ LIỆU TẠI BATCH {batch_idx}")
                print(f"- Kích thước Vocab (Model expects): {len(vocab)}")
                print(f"- Index lớn nhất trong dữ liệu (Data has): {X.max().item()}")
                print(f"- Các index bị lỗi: {X[X >= len(vocab)]}")
                # Dừng chương trình ngay lập tức để bạn xem lỗi
                import sys; sys.exit()
            
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for X, y in dev_loader:
                X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item()
                
                all_preds.append(preds)
                all_labels.append(y)
        
        # Concat & Calculate Metric
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1, acc = calculate_metrics(all_preds, all_labels)
        
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(dev_loader):.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            
            paths = {
                'model': Config.MODEL_PATH,
                'vocab': Config.VOCAB_PATH,
                'config': Config.CONFIG_PATH
            }
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                f1=best_f1,
                config_dict=Config.to_dict(),
                vocab=vocab,
                paths=paths
            )

if __name__ == "__main__":
    main()