import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

resnet34 = models.resnet34(pretrained=True)

class ResNet34Half(nn.Module):
    def __init__(self, output_shape=(28, 50)):
        super(ResNet34Half, self).__init__()
        self.resnet = nn.Sequential(*list(resnet34.children())[:6])  # get the first 6 layers
        self.in_channels = 128
        self.out_channels = 4
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.output_shape = output_shape
        
        for param in self.resnet.parameters():
            param.requires_grad = False

    def print_info(self):
        print("ResNet34Half Model:")
        print(self.resnet)

    def forward(self, x):
        x = self.resnet(x) # Forward pass through the ResNet layers, shape: (B, 128, H/4, W/4)
        x = self.up(x)  # Upsample to (B, 128, H/4, W/4)
        x = self.conv(x)  # Apply the 1x1 convolution, shape: (B, 4, H/4, W/4)
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)  # → (B, 4, 28, 50)

        return x

class HiddenImageConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=1280):
        super(HiddenImageConv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),   # 28x50 → 14x25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # 14x25 → 7x13
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),     # 7x13 → 4x6 
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1),                      # 保持 4x6, 提升通道
        )

    def forward(self, x):
        return self.encoder(x)

class HiddenImageConvPost(nn.Module):
    def __init__(self, in_channels=4, out_channels=320):
        super(HiddenImageConvPost, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    res = ResNet34Half().to('cuda')
    # res.print_info()

    image_path = "../virtual_images/default.png"

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to("cuda")  # Move to GPU if available
    with torch.no_grad():
        output = res(image_tensor)
        print(f"Output shape: {output.shape}")  # Output shape: torch.Size([1, 4, 56, 56])