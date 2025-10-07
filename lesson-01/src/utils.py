import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt

def load_dataset(base_dir):
    base_dir = Path(base_dir)

    def load_idx_images(filepath):
        with open(filepath, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number không hợp lệ: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows, cols)

    def load_idx_labels(filepath):
        with open(filepath, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number không hợp lệ: {magic}")
            return np.frombuffer(f.read(), dtype=np.uint8)

    # Load train và test
    x_train = load_idx_images(base_dir / 'train-images.idx3-ubyte')
    y_train = load_idx_labels(base_dir / 'train-labels.idx1-ubyte')
    x_test  = load_idx_images(base_dir / 't10k-images.idx3-ubyte')
    y_test  = load_idx_labels(base_dir / 't10k-labels.idx1-ubyte')

    # Gộp chung
    images = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)

    return images, labels

def show_mnist(image, label):
    """
    Hiển thị ảnh MNIST.
    
    - images: numpy array có shape (N, 28, 28)
    - labels: numpy array có shape (N,)
    - n: số lượng ảnh muốn hiển thị (mặc định = 1)
    """
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()