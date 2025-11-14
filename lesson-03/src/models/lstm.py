import torch
import torch.nn as nn

class lstm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int, 
        output_size: int,
        num_layers: int = 1, 
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim,
            padding_idx=0  
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout) 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Định nghĩa luồng dữ liệu đi qua mô hình.
        
        Args:
            x (torch.Tensor): Tensor đầu vào chứa chỉ số các token.
                              Shape: (batch_size, seq_length)
                              
        Returns:
            torch.Tensor: Logits (điểm số chưa qua softmax) cho các class.
                          Shape: (batch_size, output_size)
        """
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        last_hidden_state = h_n[-1]
        dropped = self.dropout(last_hidden_state)
        output = self.fc(dropped)
        
        return output