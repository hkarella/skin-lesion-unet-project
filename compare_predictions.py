import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset
from src.transforms import get_val_transforms
from src.models.unet import UNet


IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"


def get_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ])

    mask_paths = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_id = image_name.replace(".jpg", "")
        mask_name = image_id + "_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            raise FileNotFoundError(f"Mask not found for {image_name}")

    return image_paths, mask_paths


def load_model(path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths, mask_paths = get_image_mask_paths(IMAGE_DIR, MASK_DIR)

    _, val_imgs, _, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=42
    )

    val_dataset = SkinLesionDataset(
        val_imgs,
        val_masks,
        transform=get_val_transforms(256)
    )

    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=True)

    images, masks = next(iter(val_loader))
    images = images.to(device)

    model_epoch1 = load_model("outputs/checkpoints/unet_epoch1.pth", device)
    model_epoch3 = load_model("outputs/checkpoints/unet_epoch3.pth", device)

    with torch.no_grad():
        preds1 = torch.sigmoid(model_epoch1(images))
        preds3 = torch.sigmoid(model_epoch3(images))

        preds1 = (preds1 > 0.5).float().cpu()
        preds3 = (preds3 > 0.5).float().cpu()

    images = images.cpu()
    masks = masks.cpu()

    os.makedirs("outputs/figures", exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    titles = ["Input Image", "Ground Truth", "Epoch 1 Prediction", "Epoch 3 Prediction"]

    for i in range(3):
        img = images[i].permute(1, 2, 0)
        gt = masks[i][0]
        p1 = preds1[i][0]
        p3 = preds3[i][0]

        data = [img, gt, p1, p3]

        for j in range(4):
            if j == 0:
                axes[i, j].imshow(data[j])
            else:
                axes[i, j].imshow(data[j], cmap="gray")

            axes[i, j].set_title(titles[j], fontsize=14, fontweight="bold")
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    plt.savefig("outputs/figures/epoch_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()