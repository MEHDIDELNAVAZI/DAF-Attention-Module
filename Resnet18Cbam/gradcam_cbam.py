import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "./weights/resnet18_cbam_best.pth"
NUM_IMAGES     = 8                           # how many images to visualize
SAVE_PATH      = "./Resnet18Cbam/gradcam_cbam.png"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ─────────────────────────────────────────
# REBUILD CBAM MODEL (same as training)
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
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out)
        return x * scale.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x,  dim=1, keepdim=True)[0]
        scale   = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CBAMResNet18(nn.Module):
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
        self.cbam1   = CBAM(64,  reduction)
        self.cbam2   = CBAM(128, reduction)
        self.cbam3   = CBAM(256, reduction)
        self.cbam4   = CBAM(512, reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
# How it works:
#   1. Pick a target layer (last conv layer = layer4)
#   2. Forward pass -> save the feature map
#   3. Backward pass -> save the gradients
#   4. Weight each channel by its average gradient
#   5. Sum and ReLU -> heatmap
#   6. Resize and overlay on original image
# ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Hook to save forward activations
        self.target_layer.register_forward_hook(self._save_activation)

        # Hook to save backward gradients
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if none given
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Weight each channel by its average gradient
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam     = torch.relu(cam)  # only positive contributions

        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx


# ─────────────────────────────────────────
# OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────
def overlay_heatmap(image_np, cam, alpha=0.5):
    """
    image_np : numpy array [H, W, 3] in range 0-1
    cam      : numpy array [h, w]    in range 0-1
    returns  : blended image with heatmap
    """
    # Resize cam to image size
    cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam_resized = torch.nn.functional.interpolate(
        cam_resized,
        size=(image_np.shape[0], image_np.shape[1]),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    # Apply colormap (jet = blue->green->red)
    heatmap = cm.jet(cam_resized)[:, :, :3]  # [H, W, 3]

    # Blend heatmap with original image
    blended = alpha * heatmap + (1 - alpha) * image_np
    blended = np.clip(blended, 0, 1)
    return blended, cam_resized


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    print(f"Device: {DEVICE}")

    # ── Load model ──
    model = CBAMResNet18().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from: {WEIGHTS_PATH}")

    # ── Attach Grad-CAM to last conv layer (layer4) ──
    gradcam = GradCAM(model, target_layer=model.layer4[-1].conv2)

    # ── Load test images ──
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2023, 0.1994, 0.2010)
        )
    ])

    # Raw images for visualization (no normalize)
    vis_transform = transforms.Compose([transforms.ToTensor()])

    test_ds     = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    vis_ds      = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=vis_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # ── Generate Grad-CAM for NUM_IMAGES images ──
    print(f"\nGenerating Grad-CAM for {NUM_IMAGES} images ...")

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(12, NUM_IMAGES * 4))
    fig.suptitle("Grad-CAM — ResNet18 + CBAM", fontsize=16)

    col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight="bold")

    count = 0
    for idx, (input_tensor, label) in enumerate(test_loader):
        if count >= NUM_IMAGES:
            break

        input_tensor = input_tensor.to(DEVICE)

        # Generate heatmap
        cam, pred_idx = gradcam.generate(input_tensor)

        # Get raw image for visualization
        raw_image = vis_ds[idx][0].permute(1, 2, 0).numpy()  # [H, W, 3]
        raw_image = np.clip(raw_image, 0, 1)

        # Overlay heatmap
        overlay, cam_resized = overlay_heatmap(raw_image, cam)

        true_label = CLASSES[label.item()]
        pred_label = CLASSES[pred_idx]
        correct    = "✓" if label.item() == pred_idx else "✗"

        # Plot
        row = count

        # Original
        axes[row, 0].imshow(raw_image)
        axes[row, 0].set_ylabel(f"True: {true_label}\nPred: {pred_label} {correct}",
                                fontsize=10)
        axes[row, 0].axis("off")

        # Heatmap only
        axes[row, 1].imshow(cam_resized, cmap="jet")
        axes[row, 1].axis("off")

        # Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

        count += 1
        print(f"  [{count}/{NUM_IMAGES}] True: {true_label:12s} | Pred: {pred_label:12s} {correct}")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrad-CAM saved to: {SAVE_PATH}")
