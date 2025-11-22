# src/model.py
import torch
import torch.nn as nn

class DisasterClassifier(nn.Module):
    # Thêm tham số embedding_matrix vào __init__
    def __init__(self, vocab_size, d_model, num_classes, dropout, num_layers=2, pad_idx=0, embedding_matrix=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # --- LOGIC MỚI ---
        if embedding_matrix is not None:
            # Copy trọng số từ GloVe vào layer
            self.embedding.weight.data.copy_(embedding_matrix)
            # True: Cho phép model tinh chỉnh lại GloVe (Fine-tuning)
            # False: Giữ nguyên GloVe, không học thêm (Freeze)
            self.embedding.weight.requires_grad = True 
            print("✅ Pre-trained GloVe weights loaded into Embedding Layer")
        # -----------------
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            bidirectional=True, 
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model * 2, num_classes)

    def forward(self, inputs):
        x = self.dropout(self.embedding(inputs))
        x, _ = self.lstm(x)
        x, _ = torch.max(x, dim=1)
        logits = self.fc(self.dropout(x))
        return logits