# src/config.py
import torch
import os

class Config:
    BASE_DIR = '../'
    DATA_DIR = os.path.join(BASE_DIR, 'data/')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints/')
    
    # Paths
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')
    
    # --- THÊM DÒNG NÀY ---
    GLOVE_PATH = os.path.join(DATA_DIR, 'glove.twitter.27B.200d.txt')
    
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    VOCAB_PATH = os.path.join(CHECKPOINT_DIR, 'vocab.json')
    CONFIG_PATH = os.path.join(CHECKPOINT_DIR, 'config.json')
    
    # Params
    FREQ_THRESHOLD = 2
    BATCH_SIZE = 32
    TRAIN_DEV_SPLIT = 0.15
    
    # --- SỬA DÒNG NÀY ---
    D_MODEL = 200
    
    N_CLASSES = 2
    DROPOUT = 0.5
    N_LAYERS = 2
    EPOCHS = 20
    LR = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAD_IDX = 0 

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and k != 'to_dict' and not callable(v) and not isinstance(v, classmethod) and k != 'DEVICE'}