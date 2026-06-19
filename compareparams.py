import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────
def build_resnet18():
    model         = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


# ─────────────────────────────────────────
# SE
# ─────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.excitation(self.squeeze(x))
        return x * scale.view(x.size(0), x.size(1), 1, 1)

class SEResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        resnet       = models.resnet18(weights=None)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.se1     = SEBlock(64)
        self.se2     = SEBlock(128)
        self.se3     = SEBlock(256)
        self.se4     = SEBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ─────────────────────────────────────────
# CBAM
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
        scale = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * scale.unsqueeze(2).unsqueeze(3)

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

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

class CBAMResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        resnet       = models.resnet18(weights=None)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.cbam1   = CBAM(64)
        self.cbam2   = CBAM(128)
        self.cbam3   = CBAM(256)
        self.cbam4   = CBAM(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ─────────────────────────────────────────
# DAF
# ─────────────────────────────────────────
class ComplexityScorer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        mean   = x.mean(dim=[2, 3])
        var    = x.var(dim=[2, 3])
        stats  = torch.cat([mean, var], dim=1)
        scores = self.mlp(stats)
        return scores[:, 0], scores[:, 1]

class DAFBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.scorer      = ComplexityScorer(channels)
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        score_c, score_s = self.scorer(x)
        score_c = score_c.view(-1, 1, 1, 1)
        score_s = score_s.view(-1, 1, 1, 1)
        return x + score_c * self.channel_att(x) + score_s * self.spatial_att(x)

class DAFResNet18(nn.Module):
    def __init__(self, num_classes=10, reduction=16):
        super().__init__()
        resnet       = models.resnet18(weights=None)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.daf1    = DAFBlock(64,  reduction)
        self.daf2    = DAFBlock(128, reduction)
        self.daf3    = DAFBlock(256, reduction)
        self.daf4    = DAFBlock(512, reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.daf1(self.layer1(x))
        x = self.daf2(self.layer2(x))
        x = self.daf3(self.layer3(x))
        x = self.daf4(self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_attention_params(model, attr_names):
    total = 0
    for name in attr_names:
        if hasattr(model, name):
            total += sum(p.numel() for p in getattr(model, name).parameters())
    return total

def load_csv_results(path):
    """Read best test acc from a training CSV if it exists."""
    try:
        import csv
        best = 0.0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                val = float(row.get("test_acc", 0))
                if val > best:
                    best = val
        return best
    except Exception:
        return None

def bar(value, max_value, width=30, char="█"):
    filled = int(round(value / max_value * width))
    return char * filled + "░" * (width - filled)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    models_cfg = [
        {
            "name":        "ResNet18 (Baseline)",
            "model":       build_resnet18(),
            "att_attrs":   [],
            "csv":         "resnet18_cifar10_history.csv",
            "pth":         "resnet18_cifar10_best.pth",
        },
        {
            "name":        "ResNet18 + SE",
            "model":       SEResNet18(),
            "att_attrs":   ["se1", "se2", "se3", "se4"],
            "csv":         "resnet18_se_history.csv",
            "pth":         "resnet18_se_best.pth",
        },
        {
            "name":        "ResNet18 + CBAM",
            "model":       CBAMResNet18(),
            "att_attrs":   ["cbam1", "cbam2", "cbam3", "cbam4"],
            "csv":         "resnet18_cbam_history.csv",
            "pth":         "resnet18_cbam_best.pth",
        },
        {
            "name":        "ResNet18 + DAF",
            "model":       DAFResNet18(),
            "att_attrs":   ["daf1", "daf2", "daf3", "daf4"],
            "csv":         "resnet18_daf_history.csv",
            "pth":         "resnet18_daf_best.pth",
        },
    ]

    # ── Gather stats ──
    results = []
    for cfg in models_cfg:
        m                  = cfg["model"]
        total, trainable   = count_params(m)
        att_params         = count_attention_params(m, cfg["att_attrs"])
        base_params        = total - att_params
        best_acc           = load_csv_results(cfg["csv"])

        # try loading pth for file size
        import os
        pth_size_mb = None
        if os.path.exists(cfg["pth"]):
            pth_size_mb = os.path.getsize(cfg["pth"]) / (1024 * 1024)

        results.append({
            "name":        cfg["name"],
            "total":       total,
            "trainable":   trainable,
            "att_params":  att_params,
            "base_params": base_params,
            "best_acc":    best_acc,
            "pth_mb":      pth_size_mb,
        })

    base_total = results[0]["total"]

    W = 72
    print("\n" + "=" * W)
    print("  MODEL PARAMETER COMPARISON")
    print("=" * W)

    # ── Table header ──
    print(f"\n{'Model':<24} {'Total Params':>13} {'Attn Params':>12} {'Overhead':>10} {'Best Acc':>10}")
    print("-" * W)

    for r in results:
        overhead = ((r["total"] - base_total) / base_total * 100) if base_total else 0
        acc_str  = f"{r['best_acc']:.2f}%" if r["best_acc"] is not None else "N/A"
        oh_str   = f"+{overhead:.2f}%" if overhead >= 0 else f"{overhead:.2f}%"
        print(f"{r['name']:<24} {r['total']:>13,} {r['att_params']:>12,} {oh_str:>10} {acc_str:>10}")

    # ── Visual bars ──
    max_total = max(r["total"] for r in results)
    max_acc   = max((r["best_acc"] for r in results if r["best_acc"]), default=100)

    print("\n" + "─" * W)
    print("  TOTAL PARAMETERS  (each █ ≈ {:,} params)".format(max_total // 30))
    print("─" * W)
    for r in results:
        b = bar(r["total"], max_total)
        print(f"  {r['name']:<22} {b}  {r['total']:,}")

    if any(r["best_acc"] is not None for r in results):
        print("\n" + "─" * W)
        print("  BEST TEST ACCURACY")
        print("─" * W)
        for r in results:
            if r["best_acc"] is not None:
                b = bar(r["best_acc"], 100)
                print(f"  {r['name']:<22} {b}  {r['best_acc']:.2f}%")

    # ── Attention-only breakdown ──
    print("\n" + "─" * W)
    print("  ATTENTION MODULE OVERHEAD  (params added on top of baseline)")
    print("─" * W)
    for r in results[1:]:   # skip baseline
        added   = r["total"] - base_total
        pct     = added / base_total * 100
        b       = bar(r["att_params"], max_total, width=20)
        print(f"  {r['name']:<22} +{added:>8,} params  ({pct:.3f}% overhead)")
        print(f"  {'':22}  attn only: {b}  {r['att_params']:,}")
        print()

    # ── Model file sizes ──
    if any(r["pth_mb"] is not None for r in results):
        print("─" * W)
        print("  MODEL FILE SIZES  (.pth)")
        print("─" * W)
        max_mb = max((r["pth_mb"] for r in results if r["pth_mb"]), default=1)
        for r in results:
            if r["pth_mb"] is not None:
                b = bar(r["pth_mb"], max_mb)
                print(f"  {r['name']:<22} {b}  {r['pth_mb']:.2f} MB")

    # ── Efficiency: accuracy per 1K extra params ──
    print("\n" + "─" * W)
    print("  EFFICIENCY  (accuracy gain per 1K extra params vs baseline)")
    print("─" * W)
    base_acc = results[0]["best_acc"]
    if base_acc is not None:
        for r in results[1:]:
            if r["best_acc"] is not None:
                acc_gain    = r["best_acc"] - base_acc
                extra_k     = (r["total"] - base_total) / 1000
                efficiency  = acc_gain / extra_k if extra_k > 0 else float("inf")
                sign        = "+" if acc_gain >= 0 else ""
                print(f"  {r['name']:<22}  acc gain: {sign}{acc_gain:.2f}%   "
                      f"extra: {extra_k:.1f}K params   "
                      f"efficiency: {efficiency:.4f}%/K")
    else:
        print("  (no CSV history found — run training first)")

    print("\n" + "=" * W)
    print("  SUMMARY")
    print("=" * W)
    for r in results:
        mb_str  = f"  |  file: {r['pth_mb']:.1f} MB" if r["pth_mb"] else ""
        acc_str = f"{r['best_acc']:.2f}%" if r["best_acc"] else "N/A"
        print(f"  {r['name']:<24}  {r['total']:>11,} params  |  best acc: {acc_str}{mb_str}")
    print("=" * W + "\n")