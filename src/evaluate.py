import torch
from tqdm import tqdm
from src.metrics import dice_score, iou_score


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou += iou_score(preds, masks)

    return (
        total_loss / len(loader),
        total_dice / len(loader),
        total_iou / len(loader)
    )