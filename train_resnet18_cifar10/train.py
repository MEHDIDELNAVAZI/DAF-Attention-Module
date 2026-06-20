# ============================================================
#  ResNet18 + CIFAR-10  —  Final Baseline Training
#  Phase 1 of: CNN + Attention Research Project
# ============================================================

import os
import csv
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

# ────────────────────────────────────────────────────────────
#  0.  GOOGLE DRIVE  (uncomment if running on Colab)
# ────────────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_DIR = "/content/drive/MyDrive/ResNet18_CIFAR10/"
# os.makedirs(DRIVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
#  1.  CONFIG
# ────────────────────────────────────────────────────────────
EPOCHS        = 20
BATCH_SIZE    = 34
LR            = 0.1
PATIENCE      = 25          # early stopping
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Change to DRIVE_DIR + "..." if using Google Drive
SAVE_PATH     = "./weights/resnet18_cifar10_best.pth"
CSV_PATH      = "training_history.csv"
CHART_PATH    = "training_chart.png"

# ────────────────────────────────────────────────────────────
#  2.  DATA
# ────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ────────────────────────────────────────────────────────────
#  3.  MODEL  — ResNet18 adapted for 32×32 CIFAR-10
# ────────────────────────────────────────────────────────────
def build_resnet18_cifar():
    """
    Standard ResNet18 is designed for 224×224 (ImageNet).
    Two changes for CIFAR-10 (32×32):
      1. conv1: kernel 7→3, stride 2→1  (keeps spatial size)
      2. maxpool → Identity             (removes early downsampling)
    """
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.max(1)[1].eq(labels).sum().item()
        total      += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += inputs.size(0)
    return total_loss / total, 100.0 * correct / total

# ────────────────────────────────────────────────────────────
#  5.  CHART
# ────────────────────────────────────────────────────────────
def save_chart(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs     = [h["epoch"]      for h in history]
        train_acc  = [h["train_acc"]  for h in history]
        test_acc   = [h["test_acc"]   for h in history]
        train_loss = [h["train_loss"] for h in history]
        test_loss  = [h["test_loss"]  for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("ResNet18 on CIFAR-10 — Baseline Training", fontsize=14)

        # Accuracy
        ax1.plot(epochs, train_acc, label="Train Acc", color="#2196F3", linewidth=2)
        ax1.plot(epochs, test_acc,  label="Test Acc",  color="#4CAF50", linewidth=2)
        ax1.axhline(y=max(test_acc), color="#FF5722", linestyle="--",
                    label=f"Best Test: {max(test_acc):.2f}%")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])

        # Loss
        ax2.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
        ax2.plot(epochs, test_loss,  label="Test Loss",  color="#4CAF50", linewidth=2)
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to: {path}")
    except Exception as e:
        print(f"Chart skipped: {e}")

# ────────────────────────────────────────────────────────────
#  6.  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  ResNet18 + CIFAR-10  —  Baseline Training")
    print("=" * 55)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}  (early stop patience={PATIENCE})")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}  (CosineAnnealing)")
    print("=" * 55)

    # ── Data ──
    print("\nDownloading CIFAR-10 ...")
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train : {len(train_ds):,} images")
    print(f"  Test  : {len(test_ds):,} images")
    print(f"  Classes: {train_ds.classes}")

    # ── Model ──
    model        = build_resnet18_cifar().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model      : ResNet18 (CIFAR adapted)")
    print(f"  Parameters : {total_params:,}")

    # ── Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    # ── Training loop ──
    best_acc   = 0.0
    no_improve = 0
    history    = []
    start_time = time.time()

    print("\nStarting training ...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer)
        test_loss,  test_acc  = evaluate(
            model, test_loader,  criterion)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "test_loss":  round(test_loss,  4),
            "test_acc":   round(test_acc,   2),
            "lr":         round(scheduler.get_last_lr()[0], 6),
        })

        # Save best
        if test_acc > best_acc:
            best_acc   = test_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            tag = " ← saved ✅"
        else:
            no_improve += 1
            tag = f" (no improve {no_improve}/{PATIENCE})"

        elapsed = (time.time() - start_time) / 60
        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] "
            f"Train {train_acc:.2f}% / Test {test_acc:.2f}% "
            f"| Best: {best_acc:.2f}%{tag} | {elapsed:.1f}m"
        )

        # Early stopping
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
            break

    # ── Final summary ──
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Stopped at epoch   : {epoch}")
    print(f"  Best Test Accuracy : {best_acc:.2f}%")
    print(f"  Total Parameters   : {total_params:,}")
    print(f"  Training Time      : {total_time:.1f} min")
    print(f"  Model saved to     : {SAVE_PATH}")
    print("=" * 55)

    # ── Save CSV ──
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved to   : {CSV_PATH}")

    # ── Save chart ──
    save_chart(history, CHART_PATH)

   