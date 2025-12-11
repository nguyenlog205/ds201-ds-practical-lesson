import nltk
import string
import numpy as np
import re

from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB # Giữ Bernoulli vì nó là MVP của bản 0.742
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle

try:
    nltk.download("punkt", quiet=True)
except:
    pass

# ===== PREPROCESSING =====
punctuation = set(string.punctuation)

def preprocess_clean(text):
    try:
        tokens = word_tokenize(str(text))
        clean_tokens = []
        for token in tokens:
            lowered = token.lower()
            if lowered in punctuation:
                continue
            clean_tokens.append(lowered)
        return " ".join(clean_tokens)
    except:
        return ""

def preprocess_raw_case_sensitive(text):
    try:
        # Regex tách dấu câu (Giữ nguyên logic chiến thắng)
        text = re.sub(r'([^\w\s])', r' \1 ', str(text))
        return re.sub(r'\s+', ' ', text).strip()
    except:
        return ""

# ============================================================
#                           MODEL
# ============================================================

class model:
    def __init__(self):
        self.classifier = None
    
    def manual_oversampling(self, X, Y):
        print("--- Final Destination: Target 11k ---", flush=True)
        np.random.seed(42) 
        X = np.array(X)
        Y = np.array(Y)
        
        classes, counts = np.unique(Y, return_counts=True)
        max_count = np.max(counts)
        
        # TĂNG NHẸ LÊN 11.000: Cố gắng vắt kiệt tài nguyên server
        target_count = min(max_count, 11000)
        
        X_resampled = []
        Y_resampled = []
        
        for cls in classes:
            idx = np.where(Y == cls)[0]
            if len(idx) < target_count:
                idx_new = np.random.choice(idx, size=target_count, replace=True)
            else:
                idx_new = np.random.choice(idx, size=target_count, replace=False)
            X_resampled.extend(X[idx_new])
            Y_resampled.extend(Y[idx_new])
            
        return shuffle(X_resampled, Y_resampled, random_state=42)

    def fit(self, XTrain, YTrain):
        X_bal, Y_bal = self.manual_oversampling(XTrain, YTrain)

        # 2. PIPELINES: CẤU HÌNH 0.742 + TINH CHỈNH

        # --- NHÁNH 1: WORD EXPERT (SVM - Giữ nguyên) ---
        pipe_word = Pipeline([
            ('tfidf_word', TfidfVectorizer(
                preprocessor=preprocess_clean, tokenizer=lambda s: s.split(),
                binary=True, min_df=1, ngram_range=(1, 3), sublinear_tf=True, 
                max_features=None # Để SelectKBest lo
            )),
            ('selector', SelectKBest(chi2, k=25000)), 
            ('svm_word', LinearSVC(
                C=1.0, 
                loss='hinge', 
                class_weight='balanced', 
                dual=True, 
                intercept_scaling=1.5, 
                max_iter=3000, 
                random_state=42
            ))
        ])

        # --- NHÁNH 2: CHAR EXPERT (Logistic Regression - Mở rộng N-gram) ---
        pipe_char = Pipeline([
            ('tfidf_char', TfidfVectorizer(
                preprocessor=preprocess_clean, analyzer='char_wb',
                ngram_range=(2, 5), # <--- THAY ĐỔI: (2,5) thay vì (3,5) để bắt từ tắt 2 chữ
                min_df=2, 
                sublinear_tf=True, 
                max_features=30000
            )),
            ('lr_char', LogisticRegression(
                C=3.0, 
                solver='liblinear', 
                class_weight='balanced', 
                max_iter=300, 
                random_state=42
            ))
        ])

        # --- NHÁNH 3: RAW EXPERT (BernoulliNB - Tăng độ nhạy) ---
        pipe_raw = Pipeline([
            ('tfidf_raw', TfidfVectorizer(
                preprocessor=preprocess_raw_case_sensitive,
                lowercase=False, tokenizer=lambda s: s.split(),
                binary=True, ngram_range=(1, 4), min_df=1, sublinear_tf=True, 
                max_features=30000
            )),
            ('nb_raw', BernoulliNB(
                alpha=0.001, # <--- THAY ĐỔI: Alpha siêu nhỏ. Cực nhạy với tín hiệu hiếm.
                fit_prior=True
            ))
        ])

        # 3. VOTING (HARD)
        print("--- Training Final Ensemble ---", flush=True)
        self.classifier = VotingClassifier(
            estimators=[
                ('word_expert', pipe_word),
                ('char_expert', pipe_char),
                ('raw_expert', pipe_raw)
            ],
            voting='hard',
            n_jobs=1
        )
        
        self.classifier.fit(X_bal, Y_bal)
        print("--- Done ---", flush=True)

    def predict(self, XTest):
        if self.classifier is None:
            raise Exception("Model has not been trained yet.")
        return self.classifier.predict(XTest)