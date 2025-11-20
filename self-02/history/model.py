import nltk
import string
import numpy as np

from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle

try:
    nltk.download("punkt", quiet=True)
except:
    pass

# ===== PREPROCESSING =====

punctuation = set(string.punctuation)

def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        clean_tokens = []
        for token in tokens:
            lowered = token.lower()
            if lowered in punctuation:
                continue
            clean_tokens.append(lowered)
        return " ".join(clean_tokens)
    except:
        return ""

# ============================================================
#                           MODEL
# ============================================================

class model:
    def __init__(self):
        self.classifier = None
    
    def manual_oversampling(self, X, Y):
        np.random.seed(42) 
        X = np.array(X)
        Y = np.array(Y)
        
        classes, counts = np.unique(Y, return_counts=True)
        max_count = np.max(counts)

        target_count = min(max_count, 10000)
        
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

        pipe_word = Pipeline([
            ('tfidf_word', TfidfVectorizer(
                preprocessor=preprocess_text,
                tokenizer=lambda s: s.split(),
                binary=True,
                min_df=1,
                ngram_range=(1, 3),
                sublinear_tf=True,
                max_features=30000
            )),
            ('svm_word', LinearSVC(
                C=0.8,
                loss='hinge',
                class_weight='balanced',
                dual=True,
                intercept_scaling=1.5,
                max_iter=3000,
                random_state=42
            ))
        ])
        pipe_char = Pipeline([
            ('tfidf_char', TfidfVectorizer(
                preprocessor=preprocess_text,
                analyzer='char_wb',
                ngram_range=(3, 5),
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

        self.classifier = VotingClassifier(
            estimators=[
                ('word_expert', pipe_word),
                ('char_expert', pipe_char)
            ],
            voting='hard',
            n_jobs=-1
        )
        
        self.classifier.fit(X_bal, Y_bal)

    def predict(self, XTest):
        if self.classifier is None:
            raise Exception("Model has not been trained yet.")
        return self.classifier.predict(XTest)