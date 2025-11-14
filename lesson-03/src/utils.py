import pandas as pd
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, pad_idx):
    """
    Hàm này nhận một list các (sentence_tensor, label_tensor)
    và gộp chúng thành 2 batch tensor (sentences_batch, labels_batch)
    """
    
    sentences_list = []
    labels_list = []

    for (sentence_tensor, label_tensor) in batch:
        sentences_list.append(sentence_tensor)
        labels_list.append(label_tensor)
    labels_batch = torch.stack(labels_list)

    sentences_batch = pad_sequence(
        sentences_list, 
        batch_first=True, 
        padding_value=pad_idx
    )
    return sentences_batch, labels_batch
    

class Vocab:
    def __init__(
        self,
        path: str = r"lesson-03\dataset\uit_vsvc"
    ):
        try:
            train = pd.read_json(path + r'\UIT-VSFC-train.json')
            dev = pd.read_json(path + r'\UIT-VSFC-dev.json')
            test = pd.read_json(path + r'\UIT-VSFC-test.json')
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            print(f"Please check if the path '{path}' is correct.")
            return

        self.dataset = pd.concat([train, dev, test], ignore_index=True)
        
        # --- Label Vocabulary ---
        # Get unique labels from the 'topic' column
        labels = self.dataset['topic'].unique()

        # Create label-to-index and index-to-label mappings
        self.l2i: Dict[str, int] = {
            label: idx for idx, label in enumerate(labels)
        }
        self.i2l: Dict[int, str] = {
            idx: label for label, idx in self.l2i.items()
        }

        # --- Word Vocabulary ---
        # Add special tokens
        self.w2i: Dict[str, int] = {
            '<PAD>': 0,  # Padding token
            '<UNK>': 1   # Unknown token
        }
        
        # Build the vocabulary from the 'sentence' column
        # Note: A simple whitespace split. For Vietnamese, a real tokenizer
        # (e.g., PyVi, Underthesea) would be more robust.
        all_sentences = ' '.join(self.dataset['sentence'])
        unique_words = set(all_sentences.split())

        # Populate word-to-index mapping
        for word in unique_words:
            if word not in self.w2i:
                self.w2i[word] = len(self.w2i)  # Add new word with incrementing index

        # Create index-to-word mapping
        self.i2w: Dict[int, str] = {
            idx: word for word, idx in self.w2i.items()
        }

    @property
    def n_labels(self) -> int:
        """Returns the number of unique labels."""
        # Check if l2i was initialized (in case file loading failed)
        return len(self.l2i) if hasattr(self, 'l2i') else 0
    
    @property
    def vocab_size(self) -> int:
        """Returns the total size of the word vocabulary (including special tokens)."""
        # Check if w2i was initialized
        return len(self.w2i) if hasattr(self, 'w2i') else 0
    
    def encode_sentence(self, sentence: str) -> List[int]:
        """
        Encodes a sentence (string) into a list of word indices.
        Words not in the vocabulary are mapped to <UNK>.
        """
        words = sentence.split()
        # Use .get() with a default value (self.w2i['<UNK>']) for unknown words
        return [self.w2i.get(word, self.w2i['<UNK>']) for word in words]
    
    def decode_sentence(self, indices: List[int]) -> str:
        """
        Decodes a list of word indices back into a sentence string.
        Indices not in the vocabulary are mapped to '<??>'.
        """
        # Use .get() with a default value for unknown indices
        return ' '.join([self.i2w.get(idx, '<??>') for idx in indices])
    
    def encode_label(self, label: str) -> int:
        """Encodes a label string into its corresponding index."""
        # Use .get() to return a default value (e.g., -1) for unknown labels
        return self.l2i.get(label, -1)

    def decode_label(self, index: int) -> str:
        """Decodes a label index into its corresponding string."""
        # Use .get() for unknown indices
        return self.i2l.get(index, "<?UNKNOWN_LABEL?>")




import torch
from torch.utils.data import Dataset, DataLoader

class VsvcDataset(Dataset):
    
    def __init__(self, dataframe, vocab: Vocab):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = str(row['sentence'])
        label = str(row['topic']) 
        inputs_tensor = torch.tensor(
            self.vocab.encode_sentence(sentence), 
            dtype=torch.long
        )
        label_tensor = torch.tensor(
            self.vocab.encode_label(label), 
            dtype=torch.long
        )
        return inputs_tensor, label_tensor