# ============================================================
#  Grad-CAM — ResNet18 + DAF (Phase 4)
# ============================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────
WEIGHTS_PATH = "./weights//resnet18_daf_best.pth"
NUM_IMAGES   = 8
SAVE_PATH    = "./DAFArcitucture/gradcam_daf.png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ────────────────────────────────────────────────────────────
#  DAF MODEL  (same as training script)
# ────────────────────────────────────────────────────────────
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
        mean  = x.mean(dim=[2, 3])
        var   = x.var(dim=[2, 3])
        stats = torch.cat([mean, var], dim=1)
        scores  = self.mlp(stats)
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

# ────────────────────────────────────────────────────────────
#  GRAD-CAM
#  hook into layer4[1].conv2 — last conv before DAF and avgpool
# ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self.activation = None
        self.gradient   = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self.activation = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, inp):
        self.model.eval()
        output = self.model(inp)
        pred   = output.argmax(dim=1).item()
        conf   = output.softmax(dim=1)[0, pred].item()

        self.model.zero_grad()
        output[0, pred].backward()

        # average gradient over spatial dims → importance per channel
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]

        # weight each channel by its importance and sum
        cam = torch.relu((weights * self.activation).sum(dim=1)).squeeze()
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, pred, conf

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

# ────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])

def denorm(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * STD + MEAN, 0, 1)

def overlay_cam(cam, img_np):
    cam_up = torch.nn.functional.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0),
        size=(32, 32), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    heatmap = cm.jet(cam_up)[:, :, :3]
    return np.clip(0.45 * heatmap + 0.55 * img_np, 0, 1), cam_up

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("  Grad-CAM  —  ResNet18 + DAF")
    print("=" * 50)

    model = DAFResNet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"  Loaded: {WEIGHTS_PATH}")

    # hook into last conv layer — right before daf4 and avgpool
    gradcam = GradCAM(model, model.layer4[1].conv2)

    test_ds = torchvision.datasets.CIFAR10(
        root="data", train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    )

    # one image per class — same order as baseline and SE for fair comparison
    selected, seen = [], set()
    for img, label in test_ds:
        if label not in seen:
            seen.add(label)
            selected.append((img, label))
        if len(selected) == NUM_IMAGES:
            break

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(9, NUM_IMAGES * 2.8))
    fig.suptitle(
        "Grad-CAM — ResNet18 + DAF\n"
        "Left: original  |  Middle: heatmap  |  Right: overlay",
        fontsize=13
    )

    for row, (img_tensor, true_label) in enumerate(selected):
        inp = img_tensor.unsqueeze(0).to(DEVICE)
        cam, pred, conf = gradcam.generate(inp)
        img_np = denorm(img_tensor)
        overlay, heatmap_np = overlay_cam(cam, img_np)

        correct = pred == true_label
        color   = "green" if correct else "red"
        title   = (f"True: {CIFAR10_CLASSES[true_label]}\n"
                   f"Pred: {CIFAR10_CLASSES[pred]} ({conf*100:.0f}%)")

        axes[row, 0].imshow(img_np);                 axes[row, 0].set_title(title, fontsize=8, color=color); axes[row, 0].axis("off")
        axes[row, 1].imshow(heatmap_np, cmap="jet"); axes[row, 1].set_title("heatmap", fontsize=8);          axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay);                axes[row, 2].set_title("overlay", fontsize=8);           axes[row, 2].axis("off")

        print(f"  {'OK' if correct else 'WRONG':5s}  {CIFAR10_CLASSES[true_label]:12s} → {CIFAR10_CLASSES[pred]:12s} ({conf*100:.1f}%)")

    gradcam.remove()
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {SAVE_PATH}")
