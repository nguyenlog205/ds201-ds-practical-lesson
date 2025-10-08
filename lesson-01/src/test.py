import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate(model, X, y, batch_size=64, return_confusion_matrix=False):
    """
    Đánh giá model trên tập dữ liệu cho trước.
    Args:
        model: PyTorch model đã huấn luyện.
        X: numpy array, dữ liệu đầu vào (N, 28, 28).
        y: numpy array, nhãn (N,).
        batch_size: batch size cho DataLoader.
        return_confusion_matrix: bool, nếu True trả về confusion matrix.
    Returns:
        accuracy (float), hoặc (accuracy, confusion_matrix) nếu return_confusion_matrix=True.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    X_tensor = torch.Tensor(X).unsqueeze(1).float().to(device)
    y_tensor = torch.LongTensor(y).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    accuracy = (all_preds == all_targets).mean() * 100

    if return_confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        return accuracy, cm
    return accuracy