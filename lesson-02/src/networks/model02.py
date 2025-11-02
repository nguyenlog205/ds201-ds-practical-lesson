import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm # Thư viện cho thanh tiến trình (progress bar)

# ---------------------------------------------------------------------------
# BƯỚC 1: XÂY DỰNG CÁC KHỐI HELPER
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Một khối Convolution 2D cơ bản: Conv -> BatchNorm -> ReLU
    BatchNorm không có trong paper gốc (dùng LRN) nhưng là chuẩn hiện đại.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Inception(nn.Module):
    """
    Khối Inception theo sơ đồ bạn cung cấp.
    """
    def __init__(self, in_channels, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool):
        super(Inception, self).__init__()

        # Nhánh 1: 1x1 Conv
        self.branch1 = ConvBlock(in_channels, f_1x1, kernel_size=1)

        # Nhánh 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1),
            ConvBlock(f_3x3_r, f_3x3, kernel_size=3, padding=1)
        )

        # Nhánh 3: 1x1 Conv -> 5x5 Conv
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_5x5_r, kernel_size=1),
            ConvBlock(f_5x5_r, f_5x5, kernel_size=5, padding=2)
        )

        # Nhánh 4: 3x3 MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, f_pool, kernel_size=1)
        )

    def forward(self, x):
        # Chạy song song 4 nhánh
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate các output theo chiều channel (dim=1)
        return torch.cat([b1, b2, b3, b4], 1)

class AuxiliaryClassifier(nn.Module):
    """
    Bộ phân loại phụ (các nhánh nhỏ)
    """
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        
        # Dựa theo kích thước trong paper (giả định input 224x224)
        # Input vào nhánh này (sau Inception4a/4d) là 14x14
        # AvgPool(5,s=3) -> ((14-5)//3 + 1) = 4. Kích thước là 4x4
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.flatten = nn.Flatten()
        # Input feature: 128 channels * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 1024) 
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------------
# BƯỚC 2: XÂY DỰNG CLASS GOOGLENET CHÍNH
# ---------------------------------------------------------------------------

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        # Stem (Phần đầu)
        self.stem = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 2 x Inception
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 5 x Inception
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 2 x Inception
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        # Bộ phân loại chính
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        # Output của Inception5b là 384+384+128+128 = 1024
        self.fc = nn.Linear(1024, num_classes)
        
        # Bộ phân loại phụ
        # Output của 4a là 512, output của 4d là 528 (lưu ý nhỏ)
        # Chỉnh 4d: 112+288+64+64 = 528. OK.
        # Chỉnh 4a: 192+208+48+64 = 512. OK.
        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes) # Sửa lại in_channels=528

    def forward(self, x):
        x = self.stem(x)
        
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)
        
        x = self.inception_4a(x)
        # Chỉ kích hoạt nhánh phụ KHI TRAINING
        if self.training:
            aux1_output = self.aux1(x)
        
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training:
            aux2_output = self.aux2(x)
            
        x = self.inception_4e(x)
        x = self.pool4(x)
        
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        main_output = self.fc(x)
        
        # Trả về 3 output khi training
        if self.training:
            return main_output, aux1_output, aux2_output
        # Chỉ trả về 1 output khi evaluation (eval)
        else:
            return main_output

# ---------------------------------------------------------------------------
# BƯỚC 3: TẠO DỮ LIỆU GIẢ LẬP (MOCK DATA)
# ---------------------------------------------------------------------------
def instance_pipeline():
    print("Bắt đầu tạo dữ liệu giả lập...")
    NUM_CLASSES = 10
    INPUT_SHAPE = (3, 224, 224) # PyTorch: (Channel, Height, Width)
    BATCH_SIZE = 16
    NUM_SAMPLES_TRAIN = 100
    NUM_SAMPLES_TEST = 50

    # Dữ liệu train
    X_train = torch.randn(NUM_SAMPLES_TRAIN, *INPUT_SHAPE)
    # Nhãn là các số nguyên (cho CrossEntropyLoss)
    y_train = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES_TRAIN,)) 
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Dữ liệu test
    X_test = torch.randn(NUM_SAMPLES_TEST, *INPUT_SHAPE)
    y_test = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES_TEST,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Tạo dữ liệu xong.")

    # ---------------------------------------------------------------------------
    # BƯỚC 4: KHỞI TẠO MODEL, LOSS, OPTIMIZER
    # ---------------------------------------------------------------------------
    print("Khởi tạo mô hình...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model = GoogLeNet(num_classes=NUM_CLASSES).to(device)
    # print(model) # Bỏ comment để xem cấu trúc chi tiết

    # Hàm Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer (theo yêu cầu là Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------------------------------------------
    # BƯỚC 5: HUẤN LUYỆN VÀ ĐÁNH GIÁ
    # ---------------------------------------------------------------------------

    def evaluate(model, data_loader, device):
        """Hàm đánh giá model và in ra report (precision, recall, f1)"""
        model.eval() # Chuyển model sang chế độ evaluation
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad(): # Tắt tính toán gradient
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Khi ở model.eval(), model chỉ trả về 1 output chính
                outputs = model(inputs)
                
                # Lấy dự đoán (class có xác suất cao nhất)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Nối tất cả các batch lại
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Tạo báo cáo
        target_names = [f'Class {i}' for i in range(NUM_CLASSES)]
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=target_names,
            zero_division=0 # Tránh lỗi chia cho 0 nếu 1 class không có dự đoán
        )
        return report
