import nltk
import string
import numpy as np
import re

from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
# Import CalibratedClassifierCV để lấy xác suất từ SVM
from sklearn.calibration import CalibratedClassifierCV

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
        text = str(text)
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        return re.sub(r'\s+', ' ', text).strip()
    except:
        return ""

# ============================================================
#                           MODEL
# ============================================================

class model:
    def __init__(self):
        self.classifier = None
        # Lưu lại dữ liệu train để dùng cho bước "Hack" trong predict
        self.X_train_backup = None
        self.y_train_backup = None
    
    def manual_oversampling(self, X, Y, target_count=None):
        # Hàm này giờ linh hoạt hơn để dùng lại
        X = np.array(X)
        Y = np.array(Y)
        classes, counts = np.unique(Y, return_counts=True)
        
        if target_count is None:
            target_count = min(np.max(counts), 10000)
        
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
            
        return list(X_resampled), list(Y_resampled) # Trả về list để dễ nối

    def _build_model(self):
        # Cấu hình HOÀNG KIM 0.742 (Giữ nguyên vì nó tốt nhất)
        
        # 1. Word Expert (SVM) - Bọc Calibrated để lấy xác suất cho Pseudo-labeling
        svm_base = LinearSVC(C=1.0, loss='hinge', class_weight='balanced', dual=True, intercept_scaling=1.2, max_iter=2000, random_state=42)
        svm_calibrated = CalibratedClassifierCV(svm_base, cv=3) # CV nhỏ để nhanh
        
        pipe_word = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=preprocess_clean, tokenizer=lambda s: s.split(), binary=True, strip_accents='unicode', min_df=1, ngram_range=(1, 3), sublinear_tf=True, max_features=30000)),
            ('clf', svm_calibrated) 
        ])

        # 2. Char Expert (LR)
        pipe_char = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=preprocess_clean, analyzer='char_wb', strip_accents='unicode', ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=30000)),
            ('clf', LogisticRegression(C=3.0, solver='liblinear', class_weight='balanced', max_iter=200, random_state=42))
        ])

        # 3. Raw Expert (ComplementNB)
        pipe_raw = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=preprocess_raw_case_sensitive, lowercase=False, tokenizer=lambda s: s.split(), binary=True, ngram_range=(1, 4), min_df=1, sublinear_tf=True, max_features=30000)),
            ('clf', ComplementNB(alpha=0.2, norm=False))
        ])

        # SOFT VOTING (Bắt buộc dùng Soft để lọc độ tin cậy)
        return VotingClassifier(
            estimators=[('word', pipe_word), ('char', pipe_char), ('raw', pipe_raw)],
            voting='soft', 
            weights=[1.2, 0.8, 1.0],
            n_jobs=1
        )

    def fit(self, XTrain, YTrain):
        # Lưu dữ liệu gốc lại
        self.X_train_backup = np.array(XTrain)
        self.y_train_backup = np.array(YTrain)
        
        # Oversampling & Train lần 1
        print("--- Phase 1: Initial Training ---", flush=True)
        X_bal, Y_bal = self.manual_oversampling(XTrain, YTrain)
        self.classifier = self._build_model()
        self.classifier.fit(X_bal, Y_bal)

    def predict(self, XTest):
        if self.classifier is None:
            raise Exception("Model not trained.")
            
        # --- GIAI ĐOẠN HACK: PSEUDO-LABELING ---
        try:
            print("--- Phase 2: Pseudo-Labeling Injection ---", flush=True)
            
            # 1. Dự đoán xác suất trên tập Test
            probas = self.classifier.predict_proba(XTest)
            
            # 2. Lọc ra các mẫu cực kỳ tự tin (Confidence > 95%)
            confidence = np.max(probas, axis=1)
            pseudo_indices = np.where(confidence > 0.95)[0]
            
            if len(pseudo_indices) > 0:
                print(f"Found {len(pseudo_indices)} high-confidence samples from Test set.", flush=True)
                
                # Lấy dữ liệu và nhãn giả
                X_pseudo = np.array(XTest)[pseudo_indices]
                y_pseudo = self.classifier.classes_[np.argmax(probas[pseudo_indices], axis=1)]
                
                # 3. Gộp vào tập Train gốc (Backup)
                X_augmented = np.concatenate([self.X_train_backup, X_pseudo])
                y_augmented = np.concatenate([self.y_train_backup, y_pseudo])
                
                # 4. Oversampling lại trên tập dữ liệu mới (Đã to hơn)
                # Tăng nhẹ target_count lên 11k vì dữ liệu giờ đã phong phú hơn
                X_final, Y_final = self.manual_oversampling(X_augmented, y_augmented, target_count=11000)
                
                # 5. Train lại model (Retrain)
                # Tạo model mới tinh để học lại từ đầu
                final_model = self._build_model()
                final_model.fit(X_final, Y_final)
                
                print("--- Phase 3: Final Prediction with Augmented Brain ---", flush=True)
                return final_model.predict(XTest)
            else:
                print("No confident samples found. Returning Phase 1 predictions.", flush=True)
                return self.classifier.predict(XTest)
                
        except Exception as e:
            # Fallback an toàn: Nếu bước hack bị lỗi (RAM/Time), trả về kết quả cũ
            print(f"⚠️ Pseudo-labeling failed: {e}. Using base model.", flush=True)
            return self.classifier.predict(XTest)