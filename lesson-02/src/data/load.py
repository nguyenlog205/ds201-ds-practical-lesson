import os
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(
    data_root: str = "../../dataset",
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    Load image datasets from dataset/ and return PyTorch DataLoaders.

    Args:
        data_root: path to folder that contains `train/` and `test/` subfolders.
        img_size: square size to resize/crop images to.
        batch_size: batch size for loaders.
        val_split: fraction of training data to use as validation (0 to disable).
        shuffle: whether to shuffle training DataLoader.
        num_workers: number of workers for DataLoader.
        pin_memory: pin_memory flag for DataLoader (useful if using CUDA).

    Returns:
        (train_loader, val_loader, test_loader, class_names)
        - train_loader: DataLoader or None if train folder missing
        - val_loader: DataLoader or None if val_split == 0 or train missing
        - test_loader: DataLoader or None if test folder missing
        - class_names: list of class names (empty if no folders found)
    """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    # transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = val_loader = test_loader = None
    class_names = []

    # Load train dataset if exists
    if os.path.isdir(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        class_names = train_dataset.classes

        if val_split and val_split > 0.0:
            val_len = int(len(train_dataset) * val_split)
            train_len = len(train_dataset) - val_len
            # deterministic split
            train_subset, val_subset = random_split(
                train_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = None
    else:
        # no train folder
        train_loader = val_loader = None

    # Load test dataset if exists
    if os.path.isdir(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        # if no train folder, use test classes
        if not class_names:
            class_names = test_dataset.classes
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, class_names