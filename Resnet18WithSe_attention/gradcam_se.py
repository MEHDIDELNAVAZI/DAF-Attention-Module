# ============================================================
#  Grad-CAM Heatmap — ResNet18 + SE (Phase 2)
# ============================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────
WEIGHTS_PATH = "./weights/resnet18_se_best.pth"
NUM_IMAGES   = 8
SAVE_PATH    = "./Resnet18WithSe_attention/gradcam_se.png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ────────────────────────────────────────────────────────────
#  SE MODEL  (same as training script)
# ────────────────────────────────────────────────────────────
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
        b, c, _, _ = x.shape
        w = self.squeeze(x)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super().__init__()
        self.conv1    = nn.Conv2d(in_planes, planes, 3,
                                  stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(planes)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(planes, planes, 3,
                                  stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.se       = SEBlock(planes, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10, reduction=16):
        super().__init__()
        self.in_planes = 64
        self.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.layer1  = self._make_layer(64,  2, stride=1, reduction=reduction)
        self.layer2  = self._make_layer(128, 2, stride=2, reduction=reduction)
        self.layer3  = self._make_layer(256, 2, stride=2, reduction=reduction)
        self.layer4  = self._make_layer(512, 2, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride, reduction):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(SEBasicBlock(self.in_planes, planes,
                                       stride=s, reduction=reduction))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ────────────────────────────────────────────────────────────
#  GRAD-CAM
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
        print(self.gradient)

    def generate(self, inp):
        self.model.eval()
        output = self.model(inp)
        pred   = output.argmax(dim=1).item()
        conf   = output.softmax(dim=1)[0, pred].item()

        self.model.zero_grad()
        output[0, pred].backward()

        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activation).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam    -= cam.min()
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
    print("  Grad-CAM  —  ResNet18 + SE")
    print("=" * 50)

    model = ResNet18_SE(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"  Loaded: {WEIGHTS_PATH}")

    gradcam = GradCAM(model, model.layer4[1].conv2)

    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    )

    # one image per class — same images as baseline for fair comparison
    selected, seen = [], set()
    for img, label in test_ds:
        if label not in seen:
            seen.add(label)
            selected.append((img, label))
        if len(selected) == NUM_IMAGES:
            break

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(9, NUM_IMAGES * 2.8))
    fig.suptitle(
        "Grad-CAM — ResNet18 + SE\n"
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

        axes[row, 0].imshow(img_np);                  axes[row, 0].set_title(title, fontsize=8, color=color); axes[row, 0].axis("off")
        axes[row, 1].imshow(heatmap_np, cmap="jet");  axes[row, 1].set_title("heatmap", fontsize=8);          axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay);                 axes[row, 2].set_title("overlay", fontsize=8);           axes[row, 2].axis("off")

        print(f"  {'OK' if correct else 'WRONG':5s}  {CIFAR10_CLASSES[true_label]:12s} → {CIFAR10_CLASSES[pred]:12s} ({conf*100:.1f}%)")

    gradcam.remove()
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {SAVE_PATH}")
