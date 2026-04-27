import torch
from src.models.unet import UNet


model = UNet(in_channels=3, out_channels=1)

x = torch.randn(2, 3, 256, 256)
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)