import os
import torch
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset
from src.transforms import get_val_transforms
from src.models.unet import UNet
from src.losses import DiceLoss, BCELoss
from src.evaluate import validate


IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"
CHECKPOINT_DIR = "outputs/checkpoints"
RESULTS_PATH = "outputs/checkpoint_results.csv"


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    experiments = [
        ("Dice", "unet_epoch1.pth", DiceLoss()),
        ("Dice", "unet_epoch2.pth", DiceLoss()),
        ("Dice", "unet_epoch3.pth", DiceLoss()),
        ("BCE", "unet_bce_epoch1.pth", BCELoss()),
        ("BCE", "unet_bce_epoch2.pth", BCELoss()),
        ("BCE", "unet_bce_epoch3.pth", BCELoss()),
    ]

    os.makedirs("outputs", exist_ok=True)

    with open(RESULTS_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Loss", "Checkpoint", "Validation Loss", "Dice", "IoU"])

        for loss_name, ckpt, loss_fn in experiments:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, ckpt)

            model = UNet(in_channels=3, out_channels=1).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            val_loss, val_dice, val_iou = validate(
                model,
                val_loader,
                loss_fn,
                device
            )

            print(f"\n{loss_name} - {ckpt}")
            print("Validation Loss:", val_loss)
            print("Dice:", val_dice)
            print("IoU:", val_iou)

            writer.writerow([loss_name, ckpt, val_loss, val_dice, val_iou])

    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()