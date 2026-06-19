# ============================================================
#  Grad-CAM Heatmap — ResNet18 Baseline (Phase 1)
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
WEIGHTS_PATH = "./weights/resnet18_cifar10_best.pth"
NUM_IMAGES   = 8
SAVE_PATH    = "./train_resnet18_cifar10/gradcam_baseline.png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ────────────────────────────────────────────────────────────
#  MODEL  (same as training script)
# ────────────────────────────────────────────────────────────
def build_resnet18_cifar():
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

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

    def generate(self, inp):
        self.model.eval()
        output    = self.model(inp)
        pred      = output.argmax(dim=1).item()
        conf      = output.softmax(dim=1)[0, pred].item()

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
    print("  Grad-CAM  —  ResNet18 Baseline")
    print("=" * 50)

    model = build_resnet18_cifar().to(DEVICE)
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

    # one image per class
    selected, seen = [], set()
    for img, label in test_ds:
        if label not in seen:
            seen.add(label)
            selected.append((img, label))
        if len(selected) == NUM_IMAGES:
            break

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(9, NUM_IMAGES * 2.8))
    fig.suptitle(
        "Grad-CAM — ResNet18 Baseline\n"
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

        axes[row, 0].imshow(img_np);         axes[row, 0].set_title(title, fontsize=8, color=color); axes[row, 0].axis("off")
        axes[row, 1].imshow(heatmap_np, cmap="jet"); axes[row, 1].set_title("heatmap", fontsize=8);  axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay);        axes[row, 2].set_title("overlay", fontsize=8);           axes[row, 2].axis("off")

        print(f"  {'OK' if correct else 'WRONG':5s}  {CIFAR10_CLASSES[true_label]:12s} → {CIFAR10_CLASSES[pred]:12s} ({conf*100:.1f}%)")

    gradcam.remove()
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {SAVE_PATH}")
