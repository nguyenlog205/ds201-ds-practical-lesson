# Disaster Tweet Classification using LSTM/Bi-LSTM
> This project implements a Deep Learning solution to classify tweets from the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview) Kaggle competition, determining whether a tweet describes a real disaster or is merely using disaster-related keywords figuratively.

## Executive Summary
- **Objective**: To build a robust model capable of predicting which tweets are about a real disaster (1) and which are not (0).
- **Dataset**: Approximately 10,000 hand-labeled tweets.
- **Evaluation Metric**: F1 Score. This metric is crucial because it accounts for the potential class imbalance in the dataset by balancing Precision and Recall.

## Model Architecture and Hyperparameters
The solution utilizes a deep `Long Short-Term Memory (LSTM)` network, successfully optimized to transition from a non-learning state to strong performance.
### Hyperparameter Description
|Parameter|	Value|	Description|	Source|
|-|-|-|-|
|Architecture|Unidirectional LSTM (Ready for Bi-LSTM upgrade)|Base Model|`lstm.py`|
|Hidden Dim (`D_MODEL`)|	`512`|	Size of Embedding and LSTM Hidden State	|`main.py ` |
|Num Layers (`N_LAYERS`)|	`8`|Depth of the LSTM network	|`main.py`|
|Dropout Rate|	`0.3`	|Regularization applied after Embedding and within LSTM/FC	|`main.py`|
|Learning Rate (`LR`)|	`5e-5`	|Crucially reduced to prevent gradient explosion and saturation	|`main.py`|
|Optimizer	|`Adam`|		|`main.py`|
|Loss Function	|`nn.CrossEntropyLoss`|	Uses inverse frequency weights for imbalance handling|	`main.py`|

### Stabilization Techniques
- **Inverse Frequency Class Weights**: Weights are calculated based on the inverse frequency of the classes in the training set and applied to `nn.CrossEntropyLoss`. This technique successfully resolved the initial saturation error (Loss `~0.0000`).
- **Gradient Clipping**: `max_norm=1.0` is applied to limit the magnitude of gradients during backpropagation. This is essential for stabilizing training in the deep 8-layer LSTM.
- **Early Stopping**: Monitoring `Dev Loss` with a patience of `10` epochs ensures the best performing model checkpoint is saved.

## Data Pipeline & Preprocessing
The pipeline handles all text preparation, conversion to numerical indices, and batch creation.

### High-Level Data Flow
```txt
[Raw dataset (train.csv)] -> Vocabulary + train dataset -> Dataloader
```

### Text Preprocessing

#### Preprocessing (`preprocess_dataset`)
- **Column Concatenation**: All text columns (`text`, `keyword`, `location`) are processed and concatenated into a single tokenized column named `tokens`.
- **Text Cleaning**: Removes URLs, user mentions (@user), special escape characters, and standardizes text to lowercase.
- **Tokenization**: Converts the cleaned text into a `List[str]` (tokens).

#### Vocabulary and Data Loading
- **Vocabulary** (`Vocabulary` class): Built exclusively on the Training Data (`freq_threshold=2`) to prevent data leakage. Maps tokens to indices, using `<PAD> (0)` and `<UNK> (1)`.
- **Custom Collation** (`collate_fn`): Uses `torch.nn.utils.rnn.pad_sequence` to perform dynamic padding on each batch, ensuring all sequences are padded to the maximum length of the current batch using `PAD_IDX=2`.

## Current Results & Next Steps
> The model has successfully overcome saturation and is now learning effectively.
- **Current Dev Accuracy**: Achieved approximately `78.90%` (at Epoch 7 in the last run).
- **Best Dev Loss**: `0.4940` (at Epoch 5).
- **Kaggle** test F1-score: `0.75482`.