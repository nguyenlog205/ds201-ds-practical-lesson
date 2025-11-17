import torch
import torch.nn as nn

class lstm(nn.Module):
    """LSTM thường (không bidirectional) cho phân loại văn bản."""
    def __init__(self, vocab_size, d_model, num_classes, dropout, num_layers=2, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_size = d_model
        self.pad_idx = pad_idx # Lưu PAD_IDX

        # 1. Embedding (ĐÃ SỬA: Thêm padding_idx)
        # Giúp LSTM bỏ qua nhiễu từ token padding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)

        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=False, 
            dropout=dropout,
            batch_first=True
        )

        # 3. Dropout
        self.dropout = nn.Dropout(dropout)

        # 4. Classification head
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, inputs):
        # inputs: (batch, seq_len)
        
        # NOTE: Nếu bạn dùng Packed Sequences, bạn cần sửa dòng này
        # inputs phải là một tuple: (data, lengths)
        
        # 1. Embedding
        x = self.dropout(self.embedding(inputs))  # (batch, seq_len, d_model)

        # 2. LSTM
        _, (hidden, cell) = self.lstm(x)
        # hidden: (num_layers, batch, hidden_size)

        # 3. Lấy hidden state của layer cuối cùng (state của hướng forward, lớp cuối)
        h_last = hidden[-1] 
        
        # 4. Classifier
        logits = self.fc(self.dropout(h_last)) 

        return logits