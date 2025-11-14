import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Accuracy
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import time

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_labels: int,
    label_names: list = None # <-- Mới: List tên các nhãn (vd: ["topic_1", "topic_2"])
):
    """
    Hàm đánh giá model trên tập test.

    Args:
        model (nn.Module): Model đã huấn luyện.
        test_loader (DataLoader): DataLoader cho tập test.
        criterion (nn.Module): Hàm loss (để đo test loss).
        device (torch.device): Thiết bị (cpu hoặc cuda).
        n_labels (int): Số lượng lớp (class) của bài toán.
        label_names (list, optional): Tên của các lớp để in báo cáo chi tiết.
    """

    print("--- Bắt đầu đánh giá trên tập Test ---")
    
    # --- Khởi tạo các chỉ số đo lường ---
    # Dùng torchmetrics cho F1 (macro) và Accuracy
    f1_metric = F1Score(task="multiclass", num_classes=n_labels, average="macro").to(device)
    acc_metric = Accuracy(task="multiclass", num_classes=n_labels).to(device)
    
    total_test_loss = 0
    
    # Dùng list để lưu lại toàn bộ dự đoán và nhãn thật
    # để dùng cho classification_report của sklearn
    all_preds = []
    all_labels = []

    # --- Chuyển model sang chế độ eval ---
    model.eval()
    
    start_time = time.time()

    # --- Tắt việc tính gradient ---
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 1. Lấy dữ liệu và chuyển sang device
            # (Giả sử loader trả về (inputs, labels) đã pad)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 2. Forward pass (Đưa data qua model)
            outputs = model(inputs)

            # 3. Tính toán loss
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            # 4. Lấy dự đoán (preds)
            preds = torch.argmax(outputs, dim=1)

            # 5. Cập nhật các chỉ số của torchmetrics
            f1_metric.update(preds, labels)
            acc_metric.update(preds, labels)
            
            # 6. Lưu lại preds và labels (chuyển về CPU)
            # để dùng cho sklearn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    
    # --- Tính toán các chỉ số cuối cùng ---
    avg_test_loss = total_test_loss / len(test_loader)
    test_f1 = f1_metric.compute().item()
    test_acc = acc_metric.compute().item()

    print("\n--- 🏁 Kết quả Đánh giá trên tập Test ---")
    print(f"Thời gian đánh giá: {end_time - start_time:.2f} giây")
    print(f"Test Loss: \t{avg_test_loss:.4f}")
    print(f"Test Accuracy: \t{test_acc * 100:.2f}%")
    print(f"Test F1-Score (Macro): \t{test_f1:.4f}")
    
    # --- In báo cáo chi tiết của Sklearn ---
    print("\n📊 Báo cáo chi tiết (Classification Report):")
    if label_names:
        # Đảm bảo số lượng tên nhãn khớp với số nhãn
        if len(label_names) == n_labels:
            report = classification_report(all_labels, all_preds, target_names=label_names)
        else:
            print(f"(Lưu ý: Số lượng label_names không khớp n_labels. Sẽ dùng chỉ số 0, 1, 2...)")
            report = classification_report(all_labels, all_preds)
    else:
        report = classification_report(all_labels, all_preds)
        
    print(report)
    
    # Trả về một dict chứa các kết quả
    return {
        "test_loss": avg_test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "classification_report": report
    }