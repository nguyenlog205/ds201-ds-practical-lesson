import os
import random
import time
import json
from typing import Callable, Optional, Dict, Any, Tuple, Union, List

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.data.load import load_data
import torch.nn as nn
import torch.optim as optim


def train_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    model: Union[torch.nn.Module, Callable[..., torch.nn.Module]] = None,
    model_args: Optional[Dict[str, Any]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer_cls: Callable = optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    criterion: Optional[torch.nn.modules.loss._Loss] = None,
    device: Optional[str] = None,
    save_dir: str = "./checkpoints",
    seed: int = 42,
    log_mlflow: bool = False,
    # new: optional preprocessing callable applied to image batches BEFORE sending to device
    preprocessing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Optional[str], Optional[Dict[str, float]]]:
    """
    Generic training pipeline for any CNN model.

    - model: either an nn.Module instance or a callable that returns an nn.Module.
      If callable, it will be invoked as `model(**(model_args or {}))`.
    - preprocessing_fn: optional callable(images: Tensor) -> Tensor applied to each batch
      before moving tensors to device. Useful for custom on-the-fly preprocessing.
    - The function will also auto-adjust batch channel count to match the model's first conv.
    """
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Handle model instantiation
    if model is None:
        raise ValueError("model cannot be None")
    
    # If model is a class (callable) and not an instance, instantiate it
    if isinstance(model, type) and issubclass(model, nn.Module):
        model = model(**(model_args or {}))
    elif callable(model) and not isinstance(model, nn.Module):
        model = model(**(model_args or {}))
    
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module instance or a class/callable returning one")

    model = model.to(device)

    if train_loader is None:
        raise ValueError("train_loader cannot be None")

    # optimizer & loss
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
    criterion = criterion or nn.CrossEntropyLoss()

    best_acc = -1.0
    best_ckpt_path: Optional[str] = None
    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_precision": [],
        "val_recall": [], "val_f1": []
    }

    # optional mlflow init
    if log_mlflow:
        try:
            import mlflow
            mlflow.start_run()
            mlflow.log_param("lr", lr)
        except Exception:
            pass

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", unit="batch")
        for batch in train_iter:
            # Handle different types of batch data
            if isinstance(batch, (tuple, list)):
                images, labels = batch
            else:
                raise ValueError("Expected batch to be tuple/list of (images, labels)")

            # Ensure inputs are tensors
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            # optional preprocessing on CPU tensors (before moving to device)
            if preprocessing_fn is not None:
                images = preprocessing_fn(images)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            train_iter.set_postfix(loss=running_loss / n_samples if n_samples else 0.0)

        avg_train_loss = running_loss / n_samples if n_samples else 0.0

        # validation (if available)
        val_loss = 0.0
        precision = recall = f1 = acc = 0.0
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_targets = []
            val_loss_sum = 0.0
            val_samples = 0

            val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - val", unit="batch")
            with torch.no_grad():
                for batch in val_iter:
                    # Handle different types of batch data
                    if isinstance(batch, (tuple, list)):
                        images, labels = batch
                    else:
                        raise ValueError("Expected batch to be tuple/list of (images, labels)")

                    # Ensure inputs are tensors
                    if not isinstance(images, torch.Tensor):
                        images = torch.tensor(images)
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)

                    # optional preprocessing on CPU tensors (before moving to device)
                    if preprocessing_fn is not None:
                        images = preprocessing_fn(images)

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())

                    val_loss_sum += loss.item() * images.size(0)
                    val_samples += images.size(0)
                    val_iter.set_postfix(val_loss=val_loss_sum / val_samples if val_samples else 0.0)

            if val_samples:
                all_preds = np.concatenate(all_preds)
                all_targets = np.concatenate(all_targets)
                val_loss = val_loss_sum / val_samples
                precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
                recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
                f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
                acc = accuracy_score(all_targets, all_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} acc={acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")

        # save best model by validation accuracy
        if acc > best_acc:
            best_acc = acc
            best_ckpt_path = os.path.join(save_dir, f"model_best_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "accuracy": acc
            }, best_ckpt_path)

        # optional mlflow logging per epoch
        if log_mlflow:
            try:
                import mlflow
                mlflow.log_metric("val_accuracy", acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            except Exception:
                pass

    # final test evaluation
    test_metrics: Optional[Dict[str, float]] = None
    if test_loader is not None:
        model.eval()
        all_preds = []; all_targets = []; test_loss_sum = 0.0; test_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                # Handle different types of batch data
                if isinstance(batch, (tuple, list)):
                    images, labels = batch
                else:
                    raise ValueError("Expected batch to be tuple/list of (images, labels)")

                # Ensure inputs are tensors
                if not isinstance(images, torch.Tensor):
                    images = torch.tensor(images)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)

                # optional preprocessing on CPU tensors (before moving to device)
                if preprocessing_fn is not None:
                    images = preprocessing_fn(images)

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy()); all_targets.append(labels.cpu().numpy())
                test_loss_sum += loss.item() * images.size(0)
                test_samples += images.size(0)
        if test_samples:
            all_preds = np.concatenate(all_preds); all_targets = np.concatenate(all_targets)
            test_metrics = {
                "loss": test_loss_sum / test_samples,
                "accuracy": accuracy_score(all_targets, all_preds),
                "precision_macro": precision_score(all_targets, all_preds, average="macro", zero_division=0),
                "recall_macro": recall_score(all_targets, all_preds, average="macro", zero_division=0),
                "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
            }
            print("Test metrics:", test_metrics)

    if log_mlflow:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    return model, history, best_ckpt_path, test_metrics
