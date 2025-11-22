import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
from typing import List

# --- Preprocessing ---
def preprocess_text(text: str) -> List[str]:
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '<url>', text) 
    text = re.sub(r'@\w+', '<user>', text)
    text = re.sub(r'[^a-z0-9\s<>]', ' ', text) 
    tokens = text.split()
    return tokens

def build_vocab(texts: List[str], freq_threshold=2):
    all_tokens = [token for text in texts for token in preprocess_text(text)]
    word_counts = Counter(all_tokens)
    
    # Khởi tạo các token đặc biệt
    vocab = {"<PAD>": 0, "<UNK>": 1, "<url>": 2, "<user>": 3}
    idx = 4
    
    for word, count in word_counts.items():
        if count >= freq_threshold and word not in vocab: 
            vocab[word] = idx
            idx += 1
            
    return vocab

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, df, vocab, is_test=False):
        self.df = df
        self.vocab = vocab
        self.is_test = is_test
        self.unk_idx = vocab["<UNK>"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        tokens = preprocess_text(text)
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]
        if len(indices) == 0: indices = [self.unk_idx]
        
        tensor_x = torch.tensor(indices, dtype=torch.long)
        
        if self.is_test:
            val_id = self.df.iloc[idx]['id']
            return tensor_x, torch.tensor(val_id, dtype=torch.long) 
        else:
            label = self.df.iloc[idx]['target']
            return tensor_x, torch.tensor(label, dtype=torch.long)

def collate_fn(batch, pad_idx):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    targets = torch.stack(targets)
    
    return padded_inputs, targets