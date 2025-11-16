import torch
import torch.nn as nn

class gru(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
        """
        Khởi tạo mô hình
        :param vocab_size: Kích thước từ điển (để tạo Embedding layer)
        :param embedding_dim: Kích thước vector embedding (ví dụ: 300)
        :param hidden_size: Kích thước lớp ẩn (theo yêu cầu là 256)
        :param num_layers: Số lớp GRU (theo yêu cầu là 5)
        :param num_classes: Số lượng lớp đầu ra (ví dụ: 3 cho positive, negative, neutral)
        :param dropout: Tỷ lệ dropout
        """
        super(gru, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        hidden_last_layer = hidden[-1]
        out = self.fc(self.dropout(hidden_last_layer)) 
        return out