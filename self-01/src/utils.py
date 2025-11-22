import torch
import json
import os
from sklearn.metrics import f1_score
import numpy as np

# --- Metrics ---
def calculate_metrics(preds, labels):
    preds_cls = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    f1 = f1_score(labels, preds_cls, average='binary')
    acc = (preds_cls == labels).mean()
    return f1, acc

# --- Checkpoint Management ---
def save_checkpoint(model, optimizer, epoch, f1, config_dict, vocab, paths):
    os.makedirs(os.path.dirname(paths['model']), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1': f1
    }, paths['model'])
    
    with open(paths['vocab'], 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    with open(paths['config'], 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Saved Artifacts (Model, Vocab, Config) -> {os.path.dirname(paths['model'])}")

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_glove_embeddings(vocab, glove_path, d_model):
    """
    Đọc file GloVe và tạo ma trận embedding khớp với Vocab hiện tại.
    """
    print(f"⏳ Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # Chuyển các giá trị số thành vector
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file GloVe tại {glove_path}")
        exit()

    print(f"✅ Found {len(embeddings_index)} word vectors in GloVe.")

    # Khởi tạo ma trận ngẫu nhiên (cho các từ OOV)
    # Scale 0.6 để phương sai tương đồng với GloVe
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), d_model))
    
    hits = 0
    misses = 0

    # Map từ Vocab của project sang GloVe
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    print(f"📊 Stats: Converted {hits} words. Missed {misses} words (OOV).")
    
    return torch.tensor(embedding_matrix, dtype=torch.float)