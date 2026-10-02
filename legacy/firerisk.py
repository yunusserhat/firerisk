import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import (
    Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip,
    RandomVerticalFlip, RandomRotation, ColorJitter, GaussianBlur,
    ToTensor, Normalize
)
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import f1_score, classification_report

import timm
from timm.data import Mixup
from timm.data.random_erasing import RandomErasing
from timm.utils import ModelEmaV2, accuracy
from datasets import load_dataset

# -----------------------------
# 0. Hyper‑parameters
# -----------------------------
SEED = 100
BATCH_SIZE = 32
NUM_EPOCHS = 50
FREEZE_EPOCHS = 2      # first N epochs backbone frozen
PATIENCE = 8      # early stopping on macro‑F1
INIT_LR_HEAD = 3e-4   # LR for classifier head
INIT_LR_BACKBONE = 3e-5   # LR for backbone
WEIGHT_DECAY = 5e-2
IMG_SIZE = 224
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
CUTMIX_MINMAX = None
MIXUP_PROB = 0.8
SWITCH_PROB = 0.5
LABEL_SMOOTHING = 0.1
USE_SWA = False  # set True for Stochastic Weight Averaging
FOCAL_GAMMA = 2.0    # if 0 uses CE loss

# -----------------------------
# 1. Reproducibility helpers
# -----------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------
# 2. Dataset wrappers
# -----------------------------

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_tf = Compose([
        RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(180),
        ColorJitter(0.2, 0.2, 0.2, 0.1),
        GaussianBlur(3),
        ToTensor(),
        Normalize(IMNET_MEAN, IMNET_STD),
        RandomErasing(probability=0.25, mode="pixel")
    ])

    val_tf = Compose([
        Resize(int(IMG_SIZE * 1.14)),   # 256 for 224
        CenterCrop(IMG_SIZE),
        ToTensor(),
        Normalize(IMNET_MEAN, IMNET_STD)
    ])
    return train_tf, val_tf


class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.tf = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = self.tf(sample["image"].convert("RGB"))
        label = sample["label"]
        return image, label


# -----------------------------
# 3. Model & loss
# -----------------------------

def build_model(num_classes: int):
    backbone = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=IMG_SIZE,
        num_classes=0,
        global_pool='token'
    )
    classifier = nn.Sequential(
        nn.LayerNorm(backbone.num_features),
        nn.Linear(backbone.num_features, num_classes)
    )
    model = nn.Sequential(backbone, classifier)
    return model


class FocalLoss(nn.Module):
    """Focal Loss wrapper around CE"""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=LABEL_SMOOTHING)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# -----------------------------
# 4. Training utilities
# -----------------------------

def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    max_count = max(counts.values())
    weights = torch.tensor([max_count / counts.get(i, 1)
                           for i in range(num_classes)], dtype=torch.float)
    return weights


def train_one_epoch(model, loader, criterion, optimizer, scaler, mixup_fn, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(
            device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.tolist())
    acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100
    return acc, macro_f1, classification_report(all_labels, all_preds, output_dict=True)


# -----------------------------
# 5. Main training script
# -----------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5.1 Load dataset
    ds_name = "FireRisk"
    train_hf = load_dataset(f"yunusserhat/{ds_name}", split="train")
    val_hf = load_dataset(f"yunusserhat/{ds_name}", split="validation")
    num_classes = len(train_hf.features["label"].names)

    train_tf, val_tf = build_transforms()
    train_ds = HFImageDataset(train_hf, train_tf)
    val_ds = HFImageDataset(val_hf,  val_tf)

    # 5.2 Imbalanced handling – sampler + optional class‑weights
    class_weights = compute_class_weights(train_hf["label"], num_classes)
    sample_weights = [class_weights[l] for l in train_hf["label"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,  # MIXUP için batch boyutunu tam ve çift tut
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 5.3 Model
    model = build_model(num_classes).to(device)

    # Freeze backbone for first FREEZE_EPOCHS
    for p in model[0].parameters():
        p.requires_grad = False

    # 5.4 Loss
    if FOCAL_GAMMA > 0:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(
            device), label_smoothing=LABEL_SMOOTHING)

    # 5.5 Mixup / CutMix
    mixup_fn = Mixup(mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
                     cutmix_minmax=CUTMIX_MINMAX, prob=MIXUP_PROB,
                     switch_prob=SWITCH_PROB, label_smoothing=LABEL_SMOOTHING,
                     num_classes=num_classes) if MIXUP_PROB > 0 else None

    # 5.6 Optimizer & schedulers
    head_params = [p for p in model[1].parameters() if p.requires_grad]
    base_params = [p for p in model[0].parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {"params": base_params, "lr": INIT_LR_BACKBONE},
        {"params": head_params, "lr": INIT_LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 5.7 AMP scaler
    scaler = GradScaler()

    # 5.8 Model EMA (optional small inference boost)
    model_ema = ModelEmaV2(model, decay=0.9998,
                           device=device) if not USE_SWA else None

    # 5.9 (Optional) SWA setup
    if USE_SWA:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=INIT_LR_BACKBONE)

    # ---------------- Training loop ---------------- #
    best_f1, epochs_no_imp = 0, 0
    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            # unfreeze backbone
            for p in model[0].parameters():
                p.requires_grad = True
            optimizer.param_groups[0]["lr"] = INIT_LR_BACKBONE
            print("Backbone unfrozen – fine‑tuning entire network.")

        loss = train_one_epoch(model, train_loader,
                               criterion, optimizer, scaler, mixup_fn, device)
        scheduler.step()
        if model_ema:
            model_ema.update(model)

        # Validation
        eval_model = model_ema.module if model_ema else model
        acc, macro_f1, _ = evaluate(eval_model, val_loader, device)
        print(
            f"Epoch {epoch:02d}  |  TrainLoss {loss:.4f}  |  ValAcc {acc:.2f}  |  F1 {macro_f1:.2f}")

        improved = macro_f1 > best_f1 + 1e-3
        best_f1 = max(best_f1, macro_f1)
        epochs_no_imp = 0 if improved else epochs_no_imp + 1

        # SWA update after 75% training
        if USE_SWA and epoch > NUM_EPOCHS * 0.75:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Early stopping
        if epochs_no_imp >= PATIENCE:
            print("Early stopping triggered – no macro‑F1 improvement.")
            break

    # Save best model
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "dino2v2_firerisk_ft.pth")
    print("Model saved to", ckpt_dir / "dino2v2_firerisk_ft.pth")

    # Optionally save EMA / SWA weights
    if model_ema:
        torch.save(model_ema.ema.state_dict(), ckpt_dir /
                   "dino2v2_firerisk_ft_ema.pth")
    if USE_SWA:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        torch.save(swa_model.module.state_dict() if isinstance(swa_model, torch.nn.DataParallel) else swa_model.state_dict(),
                   ckpt_dir / "dino2v2_firerisk_ft_swa.pth")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()