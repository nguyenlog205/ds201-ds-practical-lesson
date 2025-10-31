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

def _get_model_in_channels(model: nn.Module) -> int:
    """
    Try to infer the expected input channel count of the model by
    finding the first nn.Conv2d module. Fallback to 3.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return getattr(m, "in_channels", 3)
    return 3


def _preprocess_batch_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
    """
    Convert batch of images to have target_channels.

    images: tensor (B, C, H, W) in range expected by model (already tensor).
    - If target_channels == 1 and images have 3 channels -> convert to grayscale using standard weights.
    - If target_channels == 3 and images have 1 channel -> repeat channel 3 times.
    - If channels mismatch otherwise, try simple slicing / repeating.

    Returns tensor with shape (B, target_channels, H, W).
    """
    if images.dim() != 4:
        return images
    b, c, h, w = images.shape
    if c == target_channels:
        return images

    # common conversions
    if target_channels == 1 and c == 3:
        # convert RGB to grayscale using luminosity method
        r = images[:, 0:1, :, :]
        g = images[:, 1:2, :, :]
        bch = images[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * bch
        return gray
    if target_channels == 3 and c == 1:
        return images.repeat(1, 3, 1, 1)

    # fallback: if input has more channels than needed, slice
    if c > target_channels:
        return images[:, :target_channels, :, :]

    # fallback: if input has fewer channels, repeat to match
    repeat_times = int((target_channels + c - 1) // c)
    images_rep = images.repeat(1, repeat_times, 1, 1)
    return images_rep[:, :target_channels, :, :]


def train_model(
    data_root: str,
    model: Union[torch.nn.Module, Callable[..., torch.nn.Module]],
    model_args: Optional[Dict[str, Any]] = None,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.1,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer_cls: Callable = optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    criterion: Optional[torch.nn.modules.loss._Loss] = None,
    device: Optional[str] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
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

    # instantiate model if callable
    if callable(model):
        model = model(**(model_args or {}))
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module instance or a callable returning one")

    model = model.to(device)

    # infer model expected input channels
    model_in_channels = _get_model_in_channels(model)

    # data
    train_loader, val_loader, test_loader, classes = load_data(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if train_loader is None:
        raise RuntimeError(f"Train folder not found under {data_root}/train")

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
            mlflow.log_param("batch_size", batch_size)
        except Exception:
            pass

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", unit="batch")
        for images, labels in train_iter:
            # optional preprocessing on CPU tensors (before moving to device)
            if preprocessing_fn is not None:
                images = preprocessing_fn(images)

            # ensure channel count matches model expectation
            images = _preprocess_batch_channels(images, model_in_channels)

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
                for images, labels in val_iter:
                    if preprocessing_fn is not None:
                        images = preprocessing_fn(images)

                    images = _preprocess_batch_channels(images, model_in_channels)

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
                "accuracy": acc,
                "classes": classes
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
            for images, labels in test_loader:
                if preprocessing_fn is not None:
                    images = preprocessing_fn(images)

                images = _preprocess_batch_channels(images, model_in_channels)

                images = images.to(device); labels = labels.to(device)
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
