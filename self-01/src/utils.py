import re
import string
from typing import List, Any
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd

def collate_fn(batch, PAD_ID):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    padded_data = pad_sequence(data, 
                               batch_first=True, 
                               padding_value=PAD_ID)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_data, labels

def preprocess_text(
    input_text: str
) -> List[str]:
    if not isinstance(input_text, str):
        return []
    text = input_text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)

    text = re.sub(r'‰ÛÓ|‰ÛÒ|‰Û÷|‰Ûª|‰Û¢|‰ÛÊ|‰Û_' , ' ', text)
    
    text = text.replace('#', ' # ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)


    text = re.sub(r'\s\s+', ' ', text).strip()
    tokens = text.split()
    
    return tokens

def preprocess_dataset(
    df: pd.DataFrame
):
    preprocessed_df = df.copy()

    all_columns = preprocessed_df.columns
    text_columns = [col for col in all_columns if col != 'target']

    processed_cols_data = {}

    for col_name in text_columns:
        if col_name in preprocessed_df and preprocessed_df[col_name].dtype == 'object':
            processed_cols_data[col_name] = preprocessed_df[col_name].apply(preprocess_text).tolist()


    if not processed_cols_data:
        preprocessed_df['text_tokens'] = [[]] * len(preprocessed_df)
        return preprocessed_df

    num_rows = len(preprocessed_df)
    final_preprocessed_list: List[List[str]] = []
    
    col_names_to_process = list(processed_cols_data.keys())
    
    for i in range(num_rows):
        combined_tokens: List[str] = []
        for col_name in col_names_to_process:
            tokens = processed_cols_data[col_name][i]
            combined_tokens.extend(tokens) 
            
        final_preprocessed_list.append(combined_tokens)

    preprocessed_df['tokens'] = final_preprocessed_list
    return preprocessed_df

def example_preprocess_dataset():
    df = pd.read_csv('self-01/data/train.csv')
    print(preprocess_dataset(df=df))

# example_preprocess_dataset()
# ==============================================================
# VOCABULARY
# ==============================================================
from collections import Counter
from typing import List, Dict

class Vocabulary:
    def __init__(
        self, 
        freq_threshold: int = 2
    ):
        self.freq_threshold = freq_threshold

        self.w2i: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.i2w: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        
        self.idx: int = 2

        self.PAD_IDX = self.w2i["<PAD>"]
        self.UNK_IDX = self.w2i["<UNK>"]

    def build_vocabulary(self, tokenized_texts: List[List[str]]) -> None:  
        all_words = [word for sentence in tokenized_texts for word in sentence]
        word_counts = Counter(all_words)
        
        for word, count in word_counts.items():
            if count >= self.freq_threshold and word not in self.w2i:
                self.w2i[word] = self.idx
                self.i2w[self.idx] = word
                self.idx += 1
        
        return None
    
    @property
    def n_words(self) -> int:
        return len(self.w2i)

    def encode_tokens(
        self,
        tokens_list: List[str]
    ) -> List[int]:
        """Chuyển đổi một list các tokens thành list các chỉ mục số."""
        
        # Logic tra cứu này đã đúng khi tokens_list là List[str]
        return [self.w2i.get(word, self.w2i['<UNK>']) for word in tokens_list]

class TextDataset(Dataset):
    def __init__(
        self, 
        texts_tokenized: List[List[str]], 
        labels: List[int], 
        vocab: Vocabulary
    ):
        self.texts_tokenized = texts_tokenized 
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        indices = self.vocab.encode_tokens(
            self.texts_tokenized[idx]
        )
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

def example_vocab():
    # Chuẩn bị dataset
    import pandas as pd
    df = pd.read_csv(
        r'self-01/data/train.csv'
    )
    preprocessed_dataset = preprocess_dataset(
        df=df
    )

    # Chuẩn bị bộ công cụ xây từ vựng
    vocab = Vocabulary(
        freq_threshold=2
    )
    

    vocab.build_vocabulary(
        preprocessed_dataset['tokens']
    )

    print(f"Kích thước bộ từ vựng (bao gồm <PAD>, <UNK>): {vocab.n_words}")
    '''
    # Kiểm tra ánh xạ token
    print(f"Index của 'cat': {vocab.stoi.get('cat')}") 
    print(f"Token của Index 0: {vocab.itos.get(0)}") 

    # Kiểm tra ánh xạ token OOV (Out-Of-Vocabulary)
    oov_sentence = ['the', 'new', 'dog', 'is', 'cute'] 
    indices = vocab.numericalize(oov_sentence)
    print(f"Mã hóa: {oov_sentence} -> {indices}")
    # 'new', 'is', 'cute' sẽ được ánh xạ về index của <UNK> (thường là 1)
    '''

# example_vocab()

# ===============================================================================================

def calculate_oov_rates(vocab: Vocabulary, tokenized_data: List[List[str]]):
    """
    Tính toán tỷ lệ OOV trên cấp độ từ (Word OOV Rate) và cấp độ câu (Sentence OOV Rate).

    Args:
        vocab (Vocabulary): Đối tượng Vocabulary đã được xây dựng.
        tokenized_data (List[List[str]]): Dữ liệu đã được token hóa (List of sentences).

    Returns:
        Dict: Chứa 'word_oov_rate' và 'sentence_oov_rate'.
    """
    
    total_words = 0
    unk_words = 0
    
    total_sentences = 0
    sentences_with_unk = 0
    
    unk_idx = vocab.stoi["<UNK>"]

    for sentence_tokens in tokenized_data:
        total_sentences += 1
        
        # 1. Chuyển token thành index
        indexed_sentence = vocab.to_index(sentence_tokens)
        
        # 2. Đếm từ OOV và tổng từ
        current_sentence_has_unk = False
        for index in indexed_sentence:
            total_words += 1
            if index == unk_idx:
                unk_words += 1
                current_sentence_has_unk = True
        
        # 3. Đếm câu chứa OOV
        if current_sentence_has_unk:
            sentences_with_unk += 1

    # Tính toán tỷ lệ
    word_oov_rate = (unk_words / total_words) * 100 if total_words > 0 else 0
    sentence_oov_rate = (sentences_with_unk / total_sentences) * 100 if total_sentences > 0 else 0
    
    return {
        "word_oov_rate": word_oov_rate,
        "sentence_oov_rate": sentence_oov_rate
    }

def example_calculate_oov_rate():
    import pandas as pd

    vocab = Vocabulary(freq_threshold=1)
    
    df = pd.read_csv(r'self-01/data/train.csv')
    test = pd.read_csv(r'self-01/data/test.csv')
    

    preprocessed_dataset = preprocess_dataset(df=df)
    test = preprocess_dataset(test)
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(
        preprocessed_dataset['text']
    )

    print(f"## 🛠️ Kết quả Vocabulary")
    print(f"* Kích thước từ vựng (Vocab Size): {vocab.__len__}")
    print(f"* Từ điển STOI (Snippet): {list(vocab.stoi.items())[:10]}...")
    print("-" * 30)
    oov_results = calculate_oov_rates(vocab, test['text'])

    print("## 📊 Tỷ lệ OOV trên tập Dữ liệu Kiểm tra")
    print(f"* Tỷ lệ từ OOV (Word OOV Rate): {oov_results['word_oov_rate']:.2f}%")
    print(f"* Tỷ lệ câu chứa OOV (Sentence OOV Rate): {oov_results['sentence_oov_rate']:.2f}%")

    return None

# example_calculate_oov_rate()