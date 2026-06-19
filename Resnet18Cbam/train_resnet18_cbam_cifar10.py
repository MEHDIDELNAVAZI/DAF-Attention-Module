import os
import csv
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 34
LR         = 0.1
PATIENCE   = 120
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "./weights/resnet18_cbam_best.pth"
CSV_PATH   = "./Resnet18Cbam/resnet18_cbam_history.csv"
CHART_PATH = "./Resnet18Cbam/resnet18_cbam_chart.png"

# ─────────────────────────────────────────
# CHANNEL ATTENTION
# ─────────────────────────────────────────
# Answers: "Which feature channel is important?"
#
# Feature Map [B, C, H, W]
#       |              |
#   AvgPool         MaxPool      <- two types of pooling
#       |              |
#   FC layers      FC layers     <- shared weights
#       |              |
#       +---- add ----+
#              |
#           Sigmoid              <- weight between 0 and 1
#              |
#    multiply with Feature Map   <- important channels boosted
#
# Difference from SE-Net:
#   SE-Net  -> only AvgPool
#   CBAM    -> AvgPool + MaxPool (more information)
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP — same weights used for both avg and max
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))            # AvgPool -> MLP
        max_out = self.mlp(self.max_pool(x))            # MaxPool -> MLP
        scale   = self.sigmoid(avg_out + max_out)       # add and sigmoid
        return x * scale.unsqueeze(2).unsqueeze(3)      # multiply with feature map


# ─────────────────────────────────────────
# SPATIAL ATTENTION
# ─────────────────────────────────────────
# Answers: "Where in the image is important?"
#
# Feature Map [B, C, H, W]
#       |              |
#  AvgPool(C->1)  MaxPool(C->1)  <- pool across channel dimension
#       |              |
#       concat -> [B, 2, H, W]
#              |
#         Conv 7x7               <- learns spatial importance
#              |
#           Sigmoid              <- weight for each pixel
#              |
#    multiply with Feature Map   <- important regions boosted
# ─────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out = torch.max(x,  dim=1, keepdim=True)[0] # [B, 1, H, W]
        concat  = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        scale   = self.sigmoid(self.conv(concat))        # [B, 1, H, W]
        return x * scale


# ─────────────────────────────────────────
# CBAM BLOCK
# ─────────────────────────────────────────
# Combines Channel + Spatial Attention
# Order matters: Channel first, then Spatial
#
# Feature Map
#      |
# Channel Attention   <- which feature is important?
#      |
# Spatial Attention   <- where in the image is important?
#      |
#   Output
# ─────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)  # step 1: channel attention
        x = self.spatial_att(x)  # step 2: spatial attention
        return x


# ─────────────────────────────────────────
# ResNet18 + CBAM
# ─────────────────────────────────────────
# CBAM is added after each Residual Layer:
#
#   Layer1 (64ch)  -> CBAM
#   Layer2 (128ch) -> CBAM
#   Layer3 (256ch) -> CBAM
#   Layer4 (512ch) -> CBAM
# ─────────────────────────────────────────
class CBAMResNet18(nn.Module):
    def __init__(self, num_classes=10, reduction=16):
        super(CBAMResNet18, self).__init__()

        resnet = models.resnet18(weights=None)

        # Adapt for CIFAR-10 (32x32 images instead of 224x224)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()   # remove maxpool to keep spatial size

        # ResNet layers
        self.layer1  = resnet.layer1   # output: 64 channels
        self.layer2  = resnet.layer2   # output: 128 channels
        self.layer3  = resnet.layer3   # output: 256 channels
        self.layer4  = resnet.layer4   # output: 512 channels

        # CBAM blocks — one per layer
        self.cbam1   = CBAM(64,  reduction)
        self.cbam2   = CBAM(128, reduction)
        self.cbam3   = CBAM(256, reduction)
        self.cbam4   = CBAM(512, reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layer -> CBAM attention
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────
# TRAIN / EVAL FUNCTIONS
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  ResNet18 + CBAM — Phase 3")
    print("=" * 55)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}  (CosineAnnealing)")
    print("=" * 55)

    # ── Data ──
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

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n  Train : {len(train_ds):,} images")
    print(f"  Test  : {len(test_ds):,} images")

    # ── Model ──
    model        = CBAMResNet18().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    cbam_params  = sum(p.numel() for p in model.cbam1.parameters()) + \
                   sum(p.numel() for p in model.cbam2.parameters()) + \
                   sum(p.numel() for p in model.cbam3.parameters()) + \
                   sum(p.numel() for p in model.cbam4.parameters())

    print(f"\n  Total Parameters : {total_params:,}")
    print(f"  CBAM Parameters  : {cbam_params:,}  <- added by CBAM")
    print(f"  Base Parameters  : {total_params - cbam_params:,}")

    # ── Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ──
    best_acc   = 0.0
    no_improve = 0
    history    = []
    start_time = time.time()

    print("\nStarting training ...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "test_loss":  round(test_loss,  4),
            "test_acc":   round(test_acc,   2),
        })

        if test_acc > best_acc:
            best_acc   = test_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            tag = " <- saved"
        else:
            no_improve += 1
            tag = f" (no improve {no_improve}/{PATIENCE})"

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch [{epoch:3d}/{EPOCHS}] Train {train_acc:.2f}% / Test {test_acc:.2f}% | Best: {best_acc:.2f}%{tag} | {elapsed:.1f}m")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered.")
            break

    # ── Summary ──
    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Stopped at epoch   : {epoch}")
    print(f"  Best Test Accuracy : {best_acc:.2f}%")
    print(f"  Total Parameters   : {total_params:,}")
    print(f"  CBAM Parameters    : {cbam_params:,}")
    print(f"  Training Time      : {(time.time()-start_time)/60:.1f} min")
    print(f"  Model saved to     : {SAVE_PATH}")
    print("=" * 55)

    # ── Save CSV ──
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved to: {CSV_PATH}")

    # ── Chart ──
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
        fig.suptitle("ResNet18 + CBAM on CIFAR-10", fontsize=14)

        ax1.plot(epochs, train_acc, label="Train Acc", color="#2196F3", linewidth=2)
        ax1.plot(epochs, test_acc,  label="Test Acc",  color="#4CAF50", linewidth=2)
        ax1.axhline(y=max(test_acc), color="#FF5722", linestyle="--", label=f"Best: {max(test_acc):.2f}%")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
        ax2.plot(epochs, test_loss,  label="Test Loss",  color="#4CAF50", linewidth=2)
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to: {CHART_PATH}")
    except Exception as e:
        print(f"Chart skipped: {e}")
