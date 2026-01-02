import torch
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self, is_tag=False):
        self.itos = {
            0: "<pad>", 
            1: "<unk>"
        } if not is_tag else {
            0: "<pad>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.is_tag = is_tag

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1

        for word, freq in frequencies.items():
            # Có thể đặt ngưỡng freq > 1 để lọc từ hiếm
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def encode(self, text_list):
        return [self.stoi.get(word, self.stoi.get("<unk>", 0)) for word in text_list]

    def __len__(self):
        return len(self.itos)
    
    @property
    def pad_index(self):
        return self.stoi["<pad>"]
    
    @property
    def unk_index(self):
        return self.stoi["<unk>"]
    
    @property
    def vocab_size(self):
        return len(self.itos)
    

# =================================================================
#                             DATASET
# =================================================================
class PhoNERDataset(Dataset):
    def __init__(self, words_list, tags_list, word_vocab, tag_vocab):
        self.words_list = words_list
        self.tags_list = tags_list
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, index):
        word_ids = self.word_vocab.encode(self.words_list[index])
        tag_ids = self.tag_vocab.encode(self.tags_list[index])

        return torch.tensor(word_ids), torch.tensor(tag_ids)
    
def ner_collate_fn(batch):
    (inputs, targets) = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    src_key_padding_mask = (inputs_padded == 0)
    return inputs_padded, targets_padded, src_key_padding_mask