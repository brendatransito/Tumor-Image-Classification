# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 09:47:46 2025

@author: Brenda Tránsito
"""

import os
import hashlib
import random
import numpy as np
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ============================================================
# 1. Establecer SEMILLAS para reproducibilidad
# ============================================================
def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Asegurar determinismo en operaciones de PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seeds(42)


# ============================================================
# 2. Hash MD5 (eliminación de duplicados)
# ============================================================
def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ============================================================
# 3. Cargar dataset Kaggle (Training + Testing)
# ============================================================
def load_clean_split_kaggle(base_dir):
    train_dir = os.path.join(base_dir, "Training")
    test_dir  = os.path.join(base_dir, "Testing")

    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    filepaths = []
    labels = []
    seen_hashes = set()

    print("Clases detectadas:", classes)

    for cls in classes:
        for split in [train_dir, test_dir]:

            cls_path = os.path.join(split, cls)
            if not os.path.exists(cls_path):
                continue

            imgs = (
                glob(os.path.join(cls_path, "*.jpg")) +
                glob(os.path.join(cls_path, "*.jpeg")) +
                glob(os.path.join(cls_path, "*.png"))
            )

            print(f"{cls} - {split}: {len(imgs)} imágenes")

            for img in sorted(imgs):  # Ordenar para mayor determinismo
                h = hash_image(img)

                if h not in seen_hashes:
                    seen_hashes.add(h)
                    filepaths.append(img)
                    labels.append(cls)

    print(f"\nTotal imágenes sin duplicados: {len(filepaths)}")

    if len(filepaths) == 0:
        raise ValueError("No se detectaron imágenes. Revisa BASE_DIR")

    # STRATIFIED split (determinista)
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=0.30,
        stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes


# ============================================================
# 4. Dataset personalizado
# ============================================================
class BrainTumorDataset(Dataset):
    def __init__(self, paths, labels, transform=None, class_to_idx=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# 5. DataLoaders reproducibles
# ============================================================
def prepare_dataloaders_kaggle(base_dir, batch_size=32, augment_train=True):
    """
    Crea dataloaders para:
    ✔ Train (con o sin aumentación, según augment_train)
    ✔ Validación
    ✔ Test
    """

    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = load_clean_split_kaggle(base_dir)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    print("🔢 Mapeo clase → índice:", class_to_idx)

    generator = torch.Generator()
    generator.manual_seed(42)

    # --- Transforms val/test (sin augment) ---
    val_test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # --- Transforms train (con o sin augment) ---
    if augment_train:
        print("Usando DATA AUGMENTATION en entrenamiento")
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        print("SIN data augmentation en entrenamiento")
        train_tf = val_test_tf  # misma transformación que val/test

    # --- Datasets ---
    train_set = BrainTumorDataset(X_train, y_train, train_tf, class_to_idx)
    val_set   = BrainTumorDataset(X_val,   y_val,   val_test_tf, class_to_idx)
    test_set  = BrainTumorDataset(X_test,  y_test,  val_test_tf, class_to_idx)

    # --- Dataloaders ---
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2,
                              generator=generator)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, classes
