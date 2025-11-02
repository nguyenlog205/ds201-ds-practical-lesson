import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Đây là "ResNet Block" cơ bản được hiển thị ở bên trái sơ đồ.
    Được sử dụng trong ResNet-18 và ResNet-34.
    """
    expansion = 1 # Tỷ lệ mở rộng kênh, ở BasicBlock là 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # Lớp Conv 3x3 đầu tiên
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Lớp Conv 3x3 thứ hai
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Kết nối tắt (shortcut connection)
        self.shortcut = nn.Sequential()
        
        # Nếu kích thước (spatial) hoặc số kênh (channel) thay đổi,
        # chúng ta cần một "projection shortcut" (giống như 1x1 Conv ở sơ đồ bên phải)
        # để làm cho kích thước của 'x' khớp với 'out' trước khi cộng.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Lưu lại đầu vào (identity)
        identity = x 
        
        # Đi qua các lớp chính
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Cộng với kết nối tắt (đã qua projection nếu cần)
        out += self.shortcut(identity)
        
        # Áp dụng ReLU sau khi cộng
        out = F.relu(out)
        return out

class model03(nn.Module):
    """
    Đây là class model03, triển khai kiến trúc ResNet-18.
    """
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(model03, self).__init__()
        self.in_planes = 64

        # 1. Stem (Phần đầu của mô hình)
        # Theo sơ đồ: 7x7 Conv -> Batch norm -> 3x3 Max-Pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Bốn tầng ResNet
        # ResNet-18 có [2, 2, 2, 2] khối
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 3. Head (Phần cuối)
        # Theo sơ đồ: Global Avg-Pool -> FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Avg-Pool
        self.fc = nn.Linear(512 * block.expansion, num_classes) # FC

    def _make_layer(self, block, planes, num_blocks, stride):
        """Hàm trợ giúp để xây dựng một tầng (stage) gồm nhiều khối."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # 4 Tầng ResNet
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1) # Làm phẳng (flatten) trước khi qua FC
        out = self.fc(out)
        
        return out