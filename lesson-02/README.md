# Deep Learning Practical Lesson 02: Convolutional Neural Networks

This lesson focuses on implementing and training various CNN architectures on a Vietnamese food image classification dataset.

## Project Structure

```
lesson-02/
├── dataset/                # Vietnamese food image dataset
│   ├── train/             # Training data (21 food categories)
│   └── test/              # Test data (21 food categories)
├── notebook/              # Jupyter notebooks
│   ├── report.ipynb       # Main experiment notebook
│   ├── assignments.ipynb  # Assignment solutions
│   ├── checkpoints/       # Saved model weights
│   └── logs/             # Training logs
└── src/                  # Source code
    ├── data/             # Data loading utilities
    ├── networks/         # Model architectures
    └── train_model.py    # Training pipeline
```

## Dataset

The dataset contains Vietnamese food images organized into 21 categories:
- banh-can
- banh-hoi
- banh-mi-chao
- banh-tet
- banh-trang-tron
- banh-u
- banh-uot
- bap-nuong
- bo-kho
- bo-la-lot
- bot-chien
- ca-ri
- canh-kho-qua
- canh-khoai-mo
- ga-nuong
- goi-ga
- ha-cao
- hoanh-thanh-nuoc
- pha-lau
- tau-hu
- thit-kho-trung

## Models

### 1. LeNet-5 (model01.py)
- Adapted LeNet-5 architecture for grayscale 28x28 input
- Structure:
  - Conv1: 1 → 6 channels, 5x5 kernel, padding=2
  - AvgPool1: 2x2, stride=2
  - Conv2: 6 → 16 channels, 5x5 kernel
  - AvgPool2: 2x2, stride=2
  - FC1: 16*5*5 → 120
  - FC2: 120 → 84
  - FC3: 84 → num_classes

### 2. Other Models
- model02.py: [Your model description]
- model03.py: [Your model description]
- pretrained_resnet.py: ResNet variants with transfer learning

## Usage

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Training a Model

```python
from src.data.load import load_data
from src.train_model import train_model
from src.networks.model01 import model01

# Load data
train_loader, val_loader, test_loader, classes = load_data(
    data_root="dataset",
    batch_size=32
)

# Create model instance
model = model01(num_classes=len(classes))

# Define preprocessing function (for LeNet)
def preprocessing_fn(batch):
    # Convert from [B, 3, 224, 224] -> [B, 1, 28, 28]
    gray = batch.mean(dim=1, keepdim=True)  # RGB to grayscale
    resized = torch.nn.functional.interpolate(
        gray, size=(28, 28), mode='bilinear', align_corners=False
    )
    return resized

# Train model
trained_model, history, best_ckpt_path, test_metrics = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    epochs=10,
    lr=1e-3,
    preprocessing_fn=preprocessing_fn,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### 3. Evaluating Results

The training pipeline automatically:
- Saves best model checkpoints in `notebook/checkpoints/`
- Tracks metrics:
  - Training loss
  - Validation loss
  - Accuracy
  - Precision
  - Recall
  - F1 score
- Logs results (if MLflow enabled)

## Results

Best model performances:
- LeNet-5 (model 01, train from scratch with 15 epoches): 0.1653
- GoogLeNet (model 02, train from scratch with 15 epoches): 0.5010
- ResNet18 (model 03, train from scratch with 15 epoches): 0.4223
- ResNet50 (Fine-tuned, with 15 epoches): 0.6892

## Requirements

See `requirements.txt` for full list of dependencies.

Main requirements:
- Python 3.8+
- PyTorch
- torchvision
- numpy
- sklearn
- tqdm
- mlflow (optional)
