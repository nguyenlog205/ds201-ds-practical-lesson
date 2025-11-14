import torch
import torch.nn as nn
from torchmetrics import F1Score
from tqdm.auto import tqdm
import copy
import json  # Dùng để lưu file history
import os    # Dùng để kiểm tra đường dẫn

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    n_labels,
    model_save_path,     # <-- Mới: Đường dẫn lưu model (vd: "best_model.pth")
    history_save_path, # <-- Mới: Đường dẫn lưu history (vd: "history.json")
    patience=10
):
    """
    Hàm train model, có lưu model tốt nhất và lịch sử training.
    """

    # --- Khởi tạo các biến ---
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_state = None

    # Nơi lưu trữ metrics cho tất cả các epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }

    f1_metric = F1Score(task="multiclass", num_classes=n_labels).to(device)
    model.to(device)

    print(f"--- Bắt đầu training ---")
    print(f"Lưu model tốt nhất tại: {model_save_path}")
    print(f"Lưu lịch sử training tại: {history_save_path}")
    
    # --- Vòng lặp chính qua các epoch ---
    for epoch in range(num_epochs):
        
        # --- 1. Giai đoạn Huấn luyện (Training) ---
        model.train()
        total_train_loss = 0
        
        # Giả sử loader trả về (inputs, labels)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss) # Lưu loss

        # --- 2. Giai đoạn Đánh giá (Validation / Dev) ---
        model.eval()
        total_val_loss = 0
        f1_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                f1_metric.update(preds, labels)

        avg_val_loss = total_val_loss / len(val_loader)
        # Lấy F1-score và chuyển sang số Python
        val_f1 = f1_metric.compute().item() 
        
        history["val_loss"].append(avg_val_loss) # Lưu loss
        history["val_f1"].append(val_f1)         # Lưu F1

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # --- 3. Logic Early Stopping & Lưu Model Tốt Nhất ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            
            # Lưu lại trạng thái của model
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Lưu model ra file (thường dùng .pth hoặc .pt)
            torch.save(best_model_state, model_save_path)
            print(f"🎉 New best F1: {best_val_f1:.4f}. Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}")

        # Dừng sớm
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # --- Kết thúc Training ---
    print(f"\n--- Training finished ---")
    print(f"Best Validation F1-score: {best_val_f1:.4f}")

    # Lưu file history (dạng JSON)
    try:
        with open(history_save_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history successfully saved to {history_save_path}")
    except Exception as e:
        print(f"Error saving history: {e}")

    # Tải lại trọng số của model tốt nhất trước khi trả về
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Trả về model tốt nhất và lịch sử training
    return model, history