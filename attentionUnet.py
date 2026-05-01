import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DoubleConvo(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.convo = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.ReLU()
        )
    def forward(self,x):
        return self.convo(x)

class Attention(nn.Module):
    def __init__(self,xa,ga,inter):
        super().__init__()
        self.xa = nn.Conv2d(xa,inter,kernel_size=1)
        self.ga = nn.Conv2d(ga,inter,kernel_size=1)
        self.psi = nn.Conv2d(inter,1,kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,g):
        x1 = self.xa(x)
        g1 = self.ga(g)
        psi = self.relu(x1+g1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class AttenUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConvo(1,64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConvo(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConvo(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvo(256,512)

        self.u1 = nn.ConvTranspose2d(512,256,2,2)
        self.att1 = Attention(256,256,128)
        self.convo1 = DoubleConvo(512,256)

        self.u2 = nn.ConvTranspose2d(256,128,2,2)
        self.att2 = Attention(128,128,64)
        self.convo2 = DoubleConvo(256,128)

        self.u3 = nn.ConvTranspose2d(128,64,2,2)
        self.att3 = Attention(64,64,32)
        self.convo3 = DoubleConvo(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        b = self.bottleneck(p3)

        u1 = self.u1(b)
        d3_attn = self.att1(d3,u1)
        u1 = torch.cat([u1,d3_attn],dim=1)
        u1 = self.convo1(u1)

        u2 = self.u2(u1)
        d2_attn = self.att2(d2,u2)
        u2 = torch.cat([u2,d2_attn],dim=1)
        u2 = self.convo2(u2)

        u3 = self.u3(u2)
        d1_attn = self.att3(d1,u3)
        u3 = torch.cat([u3,d1_attn],dim=1)
        u3 = self.convo3(u3)

        return torch.sigmoid(self.out(u3))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttenUnet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

img_path = "sample.jpg"

color_img = Image.open(img_path).resize((256,256))
gray_img = Image.open(img_path).convert("L").resize((256,256))

img_np = np.array(gray_img) / 255.0

img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    pred = model(img_tensor)

pred = (pred > 0.5).float().cpu().squeeze().numpy()

plt.subplot(1,2,1)
plt.title("Input")
plt.imshow(color_img)

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(pred, cmap='gray')

plt.show()