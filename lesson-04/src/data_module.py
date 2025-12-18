import pandas as pd
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from typing import List, Dict



# =======================================================================================
# DATASET
# =======================================================================================
class Vocabulary():
    def __init__(
        self,
        dataset_path: str,
        freq_threshold = 2,
    ):
        dataset = pd.read_json(dataset_path)
        print(dataset)

        english = dataset['english']
        english.apply(lambda x: [x.lower().split(' ')])
        vietnamese = dataset['vietnamese']
        
        self.en2i, self.i2en = self._build_vocab(english, min_freq=freq_threshold, language='en')
        self.vi2i, self.i2vi = self._build_vocab(vietnamese, min_freq=freq_threshold, language='vi')

    @property
    def padding_value(self):
        return 0

    @property
    def num_en_words(self):
        return len(self.en2i)
    
    @property
    def num_vi_words(self):
        return len(self.vi2i)

    def convert_en2i(self, sentence):
        return [self.en2i[word] if word in self.en2i else self.en2i['<UNK>'] for word in sentence.split()]
    
    def convert_i2en(self, idx):
        return [self.i2en[i] for i in idx]
    
    def convert_vi2i(self, sentence):
        return [self.vi2i[word] if word in self.vi2i else self.vi2i['<UNK>'] for word in sentence.split()]
    
    def convert_i2vi(self, idx):
        return [self.i2vi[i] for i in idx]


    # -----------------------------------------------------------------------------------
    # Internal function
    def _build_vocab(
        self, 
        series, 
        min_freq=2,
        language = 'en'
    ):
        word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3,
        }
        
        idx2word = {i: word for word, i in word2idx.items()}
        all_words = Counter()

        for sentence in series:
            tokens = str(sentence).lower().split()
            all_words.update(tokens)

        current_idx = 4
        for word, count in all_words.items():
            if count >= min_freq:
                if word not in word2idx:
                    word2idx[word] = current_idx
                    idx2word[current_idx] = word
                    current_idx += 1
        if language == 'en':
            print(f'Created {len(word2idx)}-word dataset in English sucessfully!')
        else:
            print(f'Created {len(word2idx)}-word dataset in Vietnamese sucessfully!')

        return word2idx, idx2word

'''
Vocabulary(
    dataset_path=r'D:\Practical-Lesson\ds210-deep-learning\lesson-04\data\small-train.json',
    freq_threshold=2
)'''
# =======================================================================================
# DATASET
# =======================================================================================
def collate_fn(
    batch: Dict,
    padding_value: int=0
):
    batch_en = []
    batch_vi = []
    for item in batch:
        batch_en.append(item[0])
        batch_vi.append(item[1])

    en_padded = pad_sequence(batch_en, batch_first=True, padding_value=padding_value)
    vi_padded = pad_sequence(batch_vi, batch_first=True, padding_value=padding_value)

    en_lengths = torch.tensor([len(x) for x in batch_en])
    vi_lengths = torch.tensor([len(x) for x in batch_vi])
    
    return {
        'encoder_input': en_padded,
        'decoder_input': vi_padded,
        'en_lengths': en_lengths,
        'vi_lengths': vi_lengths
    }

class PhoDataset(Dataset):
    def __init__(
        self, 
        dataset_path:str, 
        vocab: Vocabulary,
    ):
        self.vocab = vocab

        self.df = pd.read_json(dataset_path)
        self.english_sentences = self.df['english']
        self.vietnamese_sentences = self.df['vietnamese']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en_sentence = str(self.english_sentences.iloc[idx])
        vi_sentence = str(self.vietnamese_sentences.iloc[idx])

        en_ids = self.vocab.convert_en2i(en_sentence)
        vi_ids = self.vocab.convert_vi2i(vi_sentence)

        en_input = en_ids + [self.vocab.en2i['<EOS>']]
        vi_input = [self.vocab.vi2i['<SOS>']] + vi_ids + [self.vocab.vi2i['<EOS>']]
        return torch.tensor(en_input), torch.tensor(vi_input)