# LESSON 01

> This repository serves as lecture notes for the **Lesson 01 - Deep Learning for Data Science**. Github Repository: https://github.com/nguyenlog205/ds201-ds-practical-lesson/tree/main/lesson-01

## 🔬 Prerequisite
- Python `3.10.x` (recommend `3.10.11`)

## ⛏️ Lecture note structure
```bash
ds201-ds-practical-lesson/lesson-01
├── data/           # Dataset for the lesson
│  ├── t10k-images.idx3-ubyte  # Dataset for testing
│  ├── t10k-labels.idx1-ubyte
│  ├── train-images.idx3-ubyte # Dataset for training
│  └── train-labels.idx1-ubyte
│
├── notebook/
│   └── report.ipynb # Report of the lesson
│
├── src/             # Source code of the lesson
│   ├── ANN          # Different types of Neural Networks
│   │   ├── __init__.py
│   │   ├── CNN.py   # Convolutional Neural Network (Extra model)
│   │   ├── T1.py    # Neural Network for task 01
│   │   └── T2.py    # Neural Network for task 02
│   ├── __init__.py
│   ├── predict.py   # Pipeline for prediction
│   ├── train.py     # Pipeline for training
│   └── utils.py     # Commonly used modules/functions
│
├── .gitignore
├── lecture.ipynb    # Lesson requirements
└── README.md
```