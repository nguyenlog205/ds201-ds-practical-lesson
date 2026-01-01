import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from functools import partial

# =========================================================================================
# Vocabulary Class
# =========================================================================================
class Vocabulary:
    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.itos = {
            0: "<PAD>", 
            1: "<UNK>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2
        for sentence in sentence_list:
            for word in str(sentence).lower().split():
                frequencies[word] += 1
        
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text: str):
        tokenized_text = str(text).lower().split()
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) 
            for token in tokenized_text
        ]
    
    @property
    def pad_index(self):
        return self.stoi["<PAD>"]
    
    @property
    def unk_index(self):
        return self.stoi["<UNK>"]

    @property
    def vocab_size(self):
        return len(self.itos)

# =========================================================================================
# TextDataset
# =========================================================================================
class TextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        vocab: Vocabulary,
        label_encoder,
        text_column: str = 'review',
        label_column: str = 'domain',
        max_len: int = 256,
    ):
        self.data = pd.read_json(data_path, orient='index')
        self.vocab = vocab
        self.text_col = text_column
        self.label_col = label_column
        self.max_len = max_len
        
        self.data['label_idx'] = label_encoder.transform(self.data[self.label_col])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_raw = self.data.iloc[idx][self.text_col]
        indices = self.vocab.numericalize(text_raw)
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]

        label = self.data.iloc[idx]['label_idx']
        
        return indices, label

# =========================================================================================
# Collate Function
# =========================================================================================
def collate_fn(batch, pad_idx):
    indices_list, labels_list = zip(*batch)

    indices_tensors = [torch.tensor(seq, dtype=torch.long) for seq in indices_list]
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    inputs_padded = pad_sequence(indices_tensors, batch_first=True, padding_value=pad_idx)
    
    return inputs_padded, labels_tensor

# =========================================================================================
# 4. Cách sử dụng (Pipeline)
# =========================================================================================
# vocab = Vocabulary(...)
# vocab.build_vocabulary(...)
# label_encoder = ...

# dataset = TextDataset('data/train.json', vocab, label_encoder)
# collate_fn_p = partial(collate_fn, pad_idx=vocab.pad_index)
# loader = DataLoader(
#     dataset, 
#     batch_size=32, 
#     shuffle=True, 
#     collate_fn=collate_fn_p
# )