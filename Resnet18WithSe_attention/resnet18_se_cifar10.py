# ============================================================
#  ResNet18 + SE-Net (Channel Attention)  —  Phase 2
#  Research Project: CNN + Attention Comparison
# ============================================================
#
#  SE Block adds Channel Attention to every ResNet BasicBlock:
#
#    Feature Map [C, H, W]
#         ↓
#    Global Avg Pool → [C, 1, 1]
#         ↓
#    FC: C → C//16 → C   +   ReLU → Sigmoid
#         ↓
#    Channel weights (0–1) × Feature Map
#         ↓
#    Output: important channels boosted, weak ones suppressed
#
# ============================================================

import os
import csv
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ────────────────────────────────────────────────────────────
#  1.  CONFIG  —  same as baseline for fair comparison
# ────────────────────────────────────────────────────────────
EPOCHS        = 20
BATCH_SIZE    = 34
LR            = 0.1
PATIENCE      = 25
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH     = "resnet18_se_best.pth"
CSV_PATH      = "resnet18_se_training_history.csv"
CHART_PATH    = "resnet18_se_training_chart.png"

# ────────────────────────────────────────────────────────────
#  2.  DATA  —  identical to baseline
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
#  3.  SE BLOCK
# ────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    reduction=16 is the standard from the original SE-Net paper.

    Steps:
      1. Squeeze  : Global Avg Pool → one number per channel
      2. Excite   : Two FC layers learn channel importance weights
      3. Scale    : Multiply weights back into the feature map
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze   = nn.AdaptiveAvgPool2d(1)          # [C,H,W] → [C,1,1]
        self.excitation = nn.Sequential(
            nn.Flatten(),                                  # [C,1,1] → [C]
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()                                   # weights 0–1
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.squeeze(x)                          # [B, C, 1, 1]
        weights = self.excitation(weights)                 # [B, C]
        weights = weights.view(b, c, 1, 1)                 # [B, C, 1, 1]
        return x * weights                                 # scale channels

# ────────────────────────────────────────────────────────────
#  4.  RESNET BLOCKS WITH SE
# ────────────────────────────────────────────────────────────
class SEBasicBlock(nn.Module):
    """
    ResNet BasicBlock + SE attention after the second conv.
    Drop-in replacement for the standard BasicBlock.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # ← SE block inserted here
        self.se    = SEBlock(planes, reduction)

        # shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)           # channel attention
        out = out + self.shortcut(x) # residual
        out = self.relu(out)
        return out

# ────────────────────────────────────────────────────────────
#  5.  FULL RESNET18-SE FOR CIFAR-10
# ────────────────────────────────────────────────────────────
class ResNet18_SE(nn.Module):
    """
    ResNet18 with SE blocks, adapted for CIFAR-10 (32×32):
      - conv1: kernel 7→3, stride 2→1
      - maxpool removed (Identity)
      - all BasicBlocks replaced with SEBasicBlocks
    """
    def __init__(self, num_classes=10, reduction=16):
        super().__init__()
        self.in_planes = 64

        # stem — CIFAR adapted
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        # no maxpool for CIFAR-10

        # ResNet layers
        self.layer1  = self._make_layer(64,  2, stride=1, reduction=reduction)
        self.layer2  = self._make_layer(128, 2, stride=2, reduction=reduction)
        self.layer3  = self._make_layer(256, 2, stride=2, reduction=reduction)
        self.layer4  = self._make_layer(512, 2, stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride, reduction):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(SEBasicBlock(self.in_planes, planes,
                                       stride=s, reduction=reduction))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ────────────────────────────────────────────────────────────
#  6.  TRAIN / EVAL
# ────────────────────────────────────────────────────────────
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
#  7.  CHART
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
        fig.suptitle("ResNet18 + SE-Net on CIFAR-10 — Phase 2", fontsize=14)

        ax1.plot(epochs, train_acc, label="Train Acc", color="#2196F3", linewidth=2)
        ax1.plot(epochs, test_acc,  label="Test Acc",  color="#4CAF50", linewidth=2)
        ax1.axhline(y=max(test_acc), color="#FF5722", linestyle="--",
                    label=f"Best Test: {max(test_acc):.2f}%")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
        ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim([0, 100])

        ax2.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
        ax2.plot(epochs, test_loss,  label="Test Loss",  color="#4CAF50", linewidth=2)
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to: {path}")
    except Exception as e:
        print(f"Chart skipped: {e}")

# ────────────────────────────────────────────────────────────
#  8.  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  ResNet18 + SE-Net  —  Phase 2 Training")
    print("=" * 55)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}  (CosineAnnealing)")
    print("=" * 55)

    # ── Data ──
    print("\nLoading CIFAR-10 ...")
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ──
    model = ResNet18_SE(num_classes=10).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    se_params    = sum(p.numel() for m in model.modules()
                       if isinstance(m, SEBlock)
                       for p in m.parameters())
    base_params  = total_params - se_params

    print(f"\n  Model           : ResNet18 + SE-Net")
    print(f"  Total Params    : {total_params:,}")
    print(f"  SE Params       : {se_params:,}   ← tiny overhead")
    print(f"  Base Params     : {base_params:,}")

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
            model, test_loader, criterion)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "test_loss":  round(test_loss,  4),
            "test_acc":   round(test_acc,   2),
            "lr":         round(scheduler.get_last_lr()[0], 6),
        })

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

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Summary ──
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE — Phase 2")
    print(f"  Stopped at epoch   : {epoch}")
    print(f"  Best Test Accuracy : {best_acc:.2f}%")
    print(f"  Total Params       : {total_params:,}")
    print(f"  SE Overhead        : {se_params:,} params")
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

    # ── Compare with baseline ──
    print("\n" + "=" * 55)
    print("  HOW TO COMPARE WITH BASELINE")
    print("=" * 55)
    print("  Open both CSV files and compare best test_acc:")
    print("  training_history.csv          ← Phase 1 (ResNet18)")
    print("  resnet18_se_training_history.csv ← Phase 2 (SE)")
    print()
    print("  Expected result:")
    print("  ResNet18        → ~75-80%  (20 epochs)")
    print("  ResNet18 + SE   → ~77-82%  (20 epochs)")
    print("  SE should be ~1-2% better ✅")
    print("=" * 55)
