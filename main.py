import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.models.unet import UNet
from src.losses import DiceLoss, BCELoss
from src.train import train_one_epoch
from src.evaluate import validate


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


def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)

    LOSS_TYPE = "bce"   # use "dice" or "bce"
    num_epochs = 3

    image_paths, mask_paths = get_image_mask_paths(IMAGE_DIR, MASK_DIR)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=42
    )

    train_dataset = SkinLesionDataset(
        train_imgs,
        train_masks,
        transform=get_train_transforms(256)
    )

    val_dataset = SkinLesionDataset(
        val_imgs,
        val_masks,
        transform=get_val_transforms(256)
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Loss type:", LOSS_TYPE)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if LOSS_TYPE == "dice":
        loss_fn = DiceLoss()
    elif LOSS_TYPE == "bce":
        loss_fn = BCELoss()
    else:
        raise ValueError("LOSS_TYPE must be 'dice' or 'bce'")

    for epoch in range(num_epochs):
        print(f"\n{LOSS_TYPE.upper()} Loss - Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            loss_fn,
            device
        )

        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)
        print("Validation Dice:", val_dice)
        print("Validation IoU:", val_iou)

        checkpoint_path = f"outputs/checkpoints/unet_{LOSS_TYPE}_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()