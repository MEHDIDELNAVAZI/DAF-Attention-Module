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
SAVE_PATH  = "./weights/resnet18_daf_best.pth"
CSV_PATH   = "./DAFArcitucture/resnet18_daf_history.csv"
CHART_PATH = "./DAFArcitucture/resnet18_daf_chart.png"

# ─────────────────────────────────────────
# CHANNEL ATTENTION (same as CBAM)
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )
        return x * scale.unsqueeze(2).unsqueeze(3)


# ─────────────────────────────────────────
# SPATIAL ATTENTION (same as CBAM)
# ─────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# ─────────────────────────────────────────
# COMPLEXITY SCORER
# ─────────────────────────────────────────
# This is the INNOVATION — decides how much attention is needed
#
# For each channel in the feature map:
#   compute mean     → what is the average activation?
#   compute variance → how spread out are the activations?
#
# High variance → features are very different across spatial locations
#              → object has a clear position → Spatial Attention needed
#
# Low variance  → features are uniform across the image
#              → no clear position → Channel Attention is enough
#
# Input:  Feature Map [B, C, H, W]
# Output: [score_c, score_s]  both between 0 and 1
# ─────────────────────────────────────────
class ComplexityScorer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # MLP takes per-channel mean and variance
        # input size = channels * 2  (mean + variance for each channel)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 2, bias=False),  # 2 outputs: score_c, score_s
            nn.Sigmoid()                               # both between 0 and 1
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        # compute per-channel mean and variance across spatial dimensions
        mean = x.mean(dim=[2, 3])      # [B, C]  average per channel
        var  = x.var(dim=[2, 3])       # [B, C]  variance per channel

        # concat mean and variance → [B, C*2]
        stats = torch.cat([mean, var], dim=1)

        # MLP predicts two scores
        scores = self.mlp(stats)       # [B, 2]

        score_c = scores[:, 0]         # how much Channel Attention
        score_s = scores[:, 1]         # how much Spatial Attention

        return score_c, score_s


# ─────────────────────────────────────────
# DAF BLOCK — Dynamic Attention Fusion
# ─────────────────────────────────────────
# Full flow:
#
# Feature Map (F)
#       ↓
# ComplexityScorer → [score_c, score_s]
#       ↓
# score_c × ChannelAttention(F)  → channel refined features
# score_s × SpatialAttention(F)  → spatial refined features
#       ↓
# Output = F + channel_out + spatial_out
#
# Key difference from CBAM:
#   CBAM:  always applies both attention at weight 1.0
#   DAF:   applies each attention with a LEARNED weight
#          based on the actual variance of the feature map
# ─────────────────────────────────────────
class DAFBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.scorer      = ComplexityScorer(channels)
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        # Step 1: compute complexity scores from feature map statistics
        score_c, score_s = self.scorer(x)

        # Reshape scores for broadcasting: [B] -> [B, 1, 1, 1]
        score_c = score_c.view(-1, 1, 1, 1)
        score_s = score_s.view(-1, 1, 1, 1)

        # Step 2: apply each attention weighted by its score
        # score_c=0.9, score_s=0.1 → mostly channel attention
        # score_c=0.8, score_s=0.9 → both needed
        channel_out = score_c * self.channel_att(x)
        spatial_out = score_s * self.spatial_att(x)

        # Step 3: residual fusion
        return x + channel_out + spatial_out


# ─────────────────────────────────────────
# ResNet18 + DAF
# ─────────────────────────────────────────
class DAFResNet18(nn.Module):
    def __init__(self, num_classes=10, reduction=16):
        super().__init__()

        resnet = models.resnet18(weights=None)

        # Adapt for CIFAR-10 (32x32)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()

        # ResNet layers
        self.layer1  = resnet.layer1   # 64 channels
        self.layer2  = resnet.layer2   # 128 channels
        self.layer3  = resnet.layer3   # 256 channels
        self.layer4  = resnet.layer4   # 512 channels

        # DAF blocks — one per layer
        # Each one independently learns when to use Channel vs Spatial
        self.daf1    = DAFBlock(64,  reduction)
        self.daf2    = DAFBlock(128, reduction)
        self.daf3    = DAFBlock(256, reduction)
        self.daf4    = DAFBlock(512, reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layer → DAF attention
        x = self.daf1(self.layer1(x))
        x = self.daf2(self.layer2(x))
        x = self.daf3(self.layer3(x))
        x = self.daf4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ─────────────────────────────────────────
# TRAIN / EVAL
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

    print("=" * 60)
    print("  ResNet18 + DAF (Dynamic Attention Fusion) — Phase 4")
    print("  Innovation: variance-based attention selection")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print("=" * 60)

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
    model        = DAFResNet18().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    daf_params   = sum(p.numel() for p in model.daf1.parameters()) + \
                   sum(p.numel() for p in model.daf2.parameters()) + \
                   sum(p.numel() for p in model.daf3.parameters()) + \
                   sum(p.numel() for p in model.daf4.parameters())

    print(f"\n  Total Parameters : {total_params:,}")
    print(f"  DAF Parameters   : {daf_params:,}  <- added by DAF")
    print(f"  Base Parameters  : {total_params - daf_params:,}")

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
    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Stopped at epoch   : {epoch}")
    print(f"  Best Test Accuracy : {best_acc:.2f}%")
    print(f"  Total Parameters   : {total_params:,}")
    print(f"  DAF Parameters     : {daf_params:,}")
    print(f"  Training Time      : {(time.time()-start_time)/60:.1f} min")
    print(f"  Model saved to     : {SAVE_PATH}")
    print("=" * 60)

    # ── Save CSV ──
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved: {CSV_PATH}")

    # ── Chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_    = [h["epoch"]      for h in history]
        train_acc_ = [h["train_acc"]  for h in history]
        test_acc_  = [h["test_acc"]   for h in history]
        train_loss_= [h["train_loss"] for h in history]
        test_loss_ = [h["test_loss"]  for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("ResNet18 + DAF on CIFAR-10", fontsize=14)

        ax1.plot(epochs_, train_acc_, label="Train Acc", color="#2196F3", linewidth=2)
        ax1.plot(epochs_, test_acc_,  label="Test Acc",  color="#4CAF50", linewidth=2)
        ax1.axhline(y=max(test_acc_), color="#FF5722", linestyle="--",
                    label=f"Best: {max(test_acc_):.2f}%")
        ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_, train_loss_, label="Train Loss", color="#2196F3", linewidth=2)
        ax2.plot(epochs_, test_loss_,  label="Test Loss",  color="#4CAF50", linewidth=2)
        ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved: {CHART_PATH}")
    except Exception as e:
        print(f"Chart skipped: {e}")
