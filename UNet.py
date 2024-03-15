import torch 
import torch.nn as nn
from PIL import Image
import os 
from torch.utils.data import Dataset
import numpy as np
import sys
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision as tv
import cv2
from torchvision import transforms
import torch.nn.functional as F
from torch.nn.functional import relu
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustDataset(Dataset):
    def __init__(self, root_i, root_m):
        super().__init__()
        self.r_i = root_i
        self.r_m = root_m

        self.i_img = os.listdir(root_i)
        self.m_img = os.listdir(root_m)

        self.len_ds = max(len(self.i_img), len(self.m_img))
    def __len__(self):
        return 289
    
    def __getitem__(self, idx):
        i_img = self.i_img[idx]
        m_img = self.m_img[idx]
        i_p = os.path.join(self.r_i, i_img)
        m_p = os.path.join(self.r_m, m_img)
        
        i_img = cv2.imread(i_p, cv2.IMREAD_COLOR)
        i_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB)
        i_img = i_img.astype(np.float32)
        i_img = i_img/255.0
        i_img = cv2.resize(i_img, (400, 400), interpolation=cv2.INTER_AREA)
        
        i_img = i_img.transpose((2, 0, 1))

        m_img = cv2.imread(m_p, cv2.IMREAD_COLOR)
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        m_img = m_img.astype(np.float32)
        m_img = m_img/255.0
        m_img = cv2.resize(m_img, (400, 400), interpolation=cv2.INTER_AREA)
        
        m_img = m_img.transpose((2, 0, 1))
    #---transform---
        return i_img, m_img
    
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        out = self.outconv(xd42)

        return out
    
model = UNet(3)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
dataset = CustDataset("dataset_UNet/Image", "dataset_UNet/Mask")

loader = DataLoader(dataset, batch_size=16, )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)
loss_fn = loss_fn.to(device)

if __name__ == "__main__":
    epochs = 45
    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        i = 0
        for sample in (pbar := tqdm(loader)):
            img, mask = sample
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            
            pred = model(img)
            
            loss = loss_fn(pred, mask)

            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item
            i = i + 1 
            
            optimizer.step()
            if i % 6 == 0 and epoch % 9 == 0:
                save_image(pred, f"pred{i}_{epoch}.png")
                #save_image(img, f"img{i}_{epoch}.png")
                #save_image(mask, f"mask{i}_{epoch}.png")
            
         
