import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ─────────────────────────────────────────
# BASELINE ResNet18
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
# DAF (Dynamic Attention Fusion)
# — exact architecture from train file —
# ─────────────────────────────────────────
class ComplexityScorer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 2, bias=False),  # 2 outputs: score_c, score_s
            nn.Sigmoid()
        )
    def forward(self, x):
        mean   = x.mean(dim=[2, 3])
        var    = x.var(dim=[2, 3])
        stats  = torch.cat([mean, var], dim=1)
        scores = self.mlp(stats)
        return scores[:, 0], scores[:, 1]   # score_c, score_s


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
        channel_out = score_c * self.channel_att(x)
        spatial_out = score_s * self.spatial_att(x)
        return x + channel_out + spatial_out


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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.daf1(self.layer1(x))
        x = self.daf2(self.layer2(x))
        x = self.daf3(self.layer3(x))
        x = self.daf4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, x):
        self.model.eval()
        out  = self.model(x)
        pred = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, pred].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred


def overlay(image_np, cam, alpha=0.5, upscale=8):
    H = image_np.shape[0] * upscale
    W = image_np.shape[1] * upscale

    # Upscale original image with bicubic
    img_t  = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()
    img_up = torch.nn.functional.interpolate(
        img_t, size=(H, W), mode="bicubic", align_corners=False
    ).squeeze().permute(1, 2, 0).numpy()
    img_up = np.clip(img_up, 0, 1)

    # Upscale CAM
    cam_t  = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
    cam_up = torch.nn.functional.interpolate(
        cam_t, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    heatmap = cm.jet(cam_up)[:, :, :3]
    blended = np.clip(alpha * heatmap + (1 - alpha) * img_up, 0, 1)
    return img_up, blended


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    MODEL_CONFIGS = [
        {
            "name":    "ResNet18\n(Baseline)",
            "model":   build_resnet18(),
            "weights": "resnet18_cifar10_best.pth",
        },
        {
            "name":    "ResNet18\n+ SE",
            "model":   SEResNet18(),
            "weights": "resnet18_se_best.pth",
        },
        {
            "name":    "ResNet18\n+ CBAM",
            "model":   CBAMResNet18(),
            "weights": "resnet18_cbam_best.pth",
        },
        {
            "name":    "ResNet18\n+ DAF",
            "model":   DAFResNet18(),
            "weights": "resnet18_daf_best.pth",
        },
    ]

    NUM_IMAGES = 5

    # ── Load models ──
    loaded_models = []
    for cfg in MODEL_CONFIGS:
        try:
            state = torch.load(cfg["weights"], map_location=DEVICE)
            cfg["model"].load_state_dict(state, strict=True)
            cfg["model"] = cfg["model"].to(DEVICE).eval()
            loaded_models.append(cfg)
            print(f"✓ Loaded: {cfg['weights']}")
        except Exception as e:
            print(f"✗ Skipped {cfg['weights']}: {e}")

    num_models = len(loaded_models)
    if num_models == 0:
        raise RuntimeError("No models loaded — check your .pth paths.")

    # ── Attach Grad-CAM to layer4[-1].conv2 for every model ──
    gradcams = []
    for cfg in loaded_models:
        target = cfg["model"].layer4[-1].conv2
        gradcams.append(GradCAM(cfg["model"], target))

    # ── Data ──
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    vis_transform = transforms.Compose([transforms.ToTensor()])

    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)
    vis_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=vis_transform)

    # ── Plot ──
    # Rows: images  |  Cols: Original + one per model
    fig, axes = plt.subplots(
        NUM_IMAGES, num_models + 1,
        figsize=((num_models + 1) * 3.5, NUM_IMAGES * 3.5)
    )
    fig.suptitle("Grad-CAM Comparison — Baseline / SE / CBAM / DAF",
                 fontsize=15, fontweight="bold")

    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold")
    for col, cfg in enumerate(loaded_models):
        axes[0, col + 1].set_title(cfg["name"], fontsize=11, fontweight="bold")

    for row in range(NUM_IMAGES):
        idx        = row * 10
        raw_image  = vis_ds[idx][0].permute(1, 2, 0).numpy()
        raw_image  = np.clip(raw_image, 0, 1)
        true_label = CLASSES[vis_ds[idx][1]]
        input_t    = test_ds[idx][0].unsqueeze(0).to(DEVICE)

        raw_up, _ = overlay(raw_image, np.zeros((32, 32)), upscale=8)
        axes[row, 0].imshow(raw_up, interpolation="lanczos")
        axes[row, 0].set_ylabel(f"True: {true_label}", fontsize=9)
        axes[row, 0].axis("off")

        for col, (cfg, gc) in enumerate(zip(loaded_models, gradcams)):
            cam, pred  = gc.generate(input_t.clone())
            _, result  = overlay(raw_image, cam, upscale=8)
            pred_label = CLASSES[pred]
            correct    = "✓" if pred == vis_ds[idx][1] else "✗"

            axes[row, col + 1].imshow(result, interpolation="lanczos")
            axes[row, col + 1].set_xlabel(
                f"Pred: {pred_label} {correct}", fontsize=8)
            axes[row, col + 1].axis("off")

        print(f"Image {row + 1}/{NUM_IMAGES} done")

    plt.tight_layout()
    plt.savefig("gradcam_comparison_all_models.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved: gradcam_comparison_all_models.png")