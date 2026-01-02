# Bài thực hành 05: Mô hình Transformer

## Bài tập
- Task 01: Sentiment Classification - Phân loại miền (domain) của các đánh giá khách hàng (Review) trên bộ dữ liệu UIT-ViOCD.
- Task 02: Named Entity Recognition (NER) - Gán nhãn thực thể định danh cho dữ liệu y tế COVID-19 trên bộ dữ liệu PhoNER.
## Cấu trúc bài nộp
```t
├── data/                  # Dữ liệu gốc
├── img/                   # Ảnh lưu
├── checkpoints/           # Kết quả huấn luyện
│
├── src/              
│   ├── model01.py       # Mô hình cho bài 1
│   ├── model02.py       # Mô hình cho bài 2
│   ├── data_module1.py  # Xử lý Vocabulary, Dataset cho bài 1
│   └── data_module2.py  # Xử lý Vocabulary, Dataset cho bài 2
│
├── notebook/
│   ├── assignment.ipynb   # Bài nộp gốc
│   ├── EDA.ipynb          # Phân tích dữ liệu khám phá
│   ├── task01.ipynb       # Huấn luyện mô hình Phân loại
│   └── task02.ipynb       # Huấn luyện mô hình NER
│
└── README.md
```

## Kết quả huấn luyện
### Task 01: Domain Classification
- Dữ liệu: Các đánh giá thuộc 4 miền: `mobile`, `app`, `fashion`, `cosmetic`.
- Đánh giá: Sử dụng `Confusion Matrix` và `Classification Report` để đo lường độ chính xác trên từng miền.
### Task 02: NER
- Validation F1-Score: `~0.66`.
- Hiệu suất chi tiết:
    - Các nhãn như `GENDER`, `AGE`, `PATIENT_ID` đạt F1-score cao (>0.80).
    - Các nhãn phức tạp hơn như `ORGANIZATION`, `JOB` vẫn còn chưa tốt.