# Disaster Tweet Classification using Bi-LSTM
> **Competition Link**: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview) 


## Executive Summary
> This project implements a robust Deep Learning solution to classify disaster-related tweets. Moving beyond standard approaches, this solution leverages Transfer Learning using GloVe (Twitter) pre-trained embeddings combined with a Bidirectional LSTM architecture.
- **Objective**: To build a robust model capable of predicting which tweets are about a real disaster (1) and which are not (0).
- **Dataset**: Approximately 10,000 hand-labeled tweets.
- **Evaluation Metric**: F1 Score. This metric is crucial because it accounts for the potential class imbalance in the dataset by balancing Precision and Recall.

## Model Architecture and Hyperparameters
The model architecture uses a **Bidirectional LSTM** followed by **Global Max Pooling**, which effectively captures the most salient features (keywords like "`kill`", "`accident`") regardless of their position in the tweet.
### Hyperparameter Description
|Component|	Value|	Description|
|-|-|-|
|Architecture|Bi-LSTM|Bidirectional Long Short-Term Memory|
|Embedding|GloVe 200d | Pre-trained on 2B tweets. Fine-tuned during training.|
|Hidden Dim| 200|Matches embedding dimension for consistency.|
|Num Layers|2||
|Dropout|0.5|
|Optimizer|Adam|Standard optimizer with `lr=1e-4`.|
|Loss Function|CrossEntropyLoss | Weighted to handle class imbalance.|

### Stabilization Configuration
- **Inverse Frequency Class Weights**: Weights are calculated based on the inverse frequency of the classes in the training set and applied to `nn.CrossEntropyLoss`. This technique successfully resolved the initial saturation error (Loss `~0.0000`).
- **Gradient Clipping**: `max_norm=1.0` is applied to limit the magnitude of gradients during backpropagation. This is essential for stabilizing training in the deep 8-layer LSTM.
- **Early Stopping**: Monitoring `Dev Loss` with a patience of `10` epochs ensures the best performing model checkpoint is saved.

### Engineering Optimizations (MLOps)
**Artifacts Management**: Ensures 100% reproducibility by saving a trio of artifacts for every checkpoint:
- ` best_model.pth` (Weights)
- `vocab.json` (Token mapping)
- `config.json` (Hyperparameters)

## Data Pipeline & Preprocessing
The project features a modular pipeline located in `src/`.
```txt
graph LR
    A[Raw CSV] --> B(Preprocessing)
    B --> C{Check GloVe Cache}
    C -- No --> D[Parse .txt & Save .pt]
    C -- Yes --> E[Load .pt Tensor]
    E --> F[Embedding Layer]
    B --> G[DataLoader]
```

### Preprocessing Flow
1. **Cleaning**: `URLs` -> `<url>`, `Usernames` -> `<user>`.
2. **Vocabulary**: Built dynamically from training data, strictly handling OOV (Out-Of-Vocabulary) tokens.
3. **Vector Mapping**: Maps tokens to GloVe vectors. Words not in GloVe are initialized with a normal distribution matching GloVe's variance.

## Competition Execution

### Prerequisites
```txt
pip install torch pandas numpy scikit-learn
```
### Data setup
**Download URL**: [glove.twitter.27B.200d.txt](https://www.kaggle.com/datasets/fullmetal26/glovetwitter27b100dtxt).

After download completely, unzip it and then move `glove.twitter.27B.200d.txt` to `self-01/data/`.

```txt
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── glove.twitter.27B.200d.txt
...
```
### Training and Inferencing
#### Training
- Module: `src/train.py`.
- This script will automatically build the vocab, cache the embeddings, and start training.
```bash
cd src
python train.py
```

#### Inferencing
- Module: `src/inference.py`.
- To generate submission.csv using the best saved model (automatically loads the correct vocab/config).
```bash
cd src
python inference.py
```

## Current Results & Next Steps
> The model has successfully overcome saturation and is now learning effectively.
- **Current Dev Accuracy**: Achieved approximately `0.8117` (at Epoch 9 in the last run).
- **Kaggle** test F1-score: `0.80416`.