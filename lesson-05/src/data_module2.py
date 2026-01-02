import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def build(self, sentences):
        freqs = Counter()
        for sent in sentences:
            for word in sent:
                freqs[word] += 1
        for word, count in freqs.items():
            if count >= self.freq_threshold:
                self.stoi[word] = len(self.itos)
                self.itos[len(self.itos)] = word

    def numericalize(self, words):
        return [self.stoi.get(w, self.stoi["<UNK>"]) for w in words]

class NERDataset(Dataset):
    def __init__(self, sentences, tags, vocab, tag_map):
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab
        self.tag_map = tag_map 

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]

        word_indices = self.vocab.numericalize(words)

        tag_indices = [self.tag_map[t] for t in tags]

        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

def ner_collate_fn(batch, pad_idx_text=0, pad_idx_label=-100):
    text_list, label_list = zip(*batch)
    padded_text = pad_sequence(text_list, batch_first=True, padding_value=pad_idx_text)
    padded_labels = pad_sequence(label_list, batch_first=True, padding_value=pad_idx_label)
    return padded_text, padded_labels