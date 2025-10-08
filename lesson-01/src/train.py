import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from src.ANN.CNN import CNN
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import load_dataset

def train(
       X_data, 
       y_data, 
       model_class,
       LEARNING_RATE = 0.001,
       BATCH_SIZE = 64,
       EPOCHS = 10,
       MLFLOW_EXPERIMENT_NAME = "MNIST_Classification"
):
    """
    Hàm chính để thực hiện training và tracking với MLflow từ dữ liệu đã được tải sẵn.
    """
    # --- 2. Chuẩn bị dữ liệu ---
    print("Đang chuẩn bị và chia dữ liệu...")
    X_train, X_val, y_train, y_val = train_test_split(
       X_data, y_data, 
       test_size=0.2,
       random_state=42,
       stratify=y_data
    )

    X_train_tensor = torch.Tensor(X_train).unsqueeze(1).float()
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.Tensor(X_val).unsqueeze(1).float()
    y_val_tensor = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Chuẩn bị dữ liệu thành công!")

    # --- 3. Bắt đầu phiên làm việc với MLflow ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        print(f"Bắt đầu MLflow run: {run.info.run_id}")
        
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("model_class", model_class.__name__)

        # --- 4. Khởi tạo model, optimizer và loss function ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()

        # --- 5. Vòng lặp Training & Validation ---
        for epoch in range(EPOCHS):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader.dataset)
            val_accuracy = 100. * correct / len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", val_accuracy, step=epoch)

        # --- 6. Log model đã huấn luyện vào MLflow ---
        print("Training hoàn tất. Đang lưu model vào MLflow...")
        mlflow.pytorch.log_model(model, "model")
        print("Lưu model thành công!")

def train_pipeline():
    # Load dataset
    DATA_DIR = '../data'
    dataset = load_dataset(DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        dataset[0], dataset[1], 
        test_size=0.1,
        random_state=42
    )
    
    from ANN.CNN import CNN
    train(X_data=X_train,
          y_data=y_train,
          model_class=CNN,
          LEARNING_RATE=0.001,
          BATCH_SIZE=64,
          EPOCHS=10,
          MLFLOW_EXPERIMENT_NAME=CNN.__name__
    )