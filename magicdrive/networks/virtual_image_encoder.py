import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import argparse
from typing import Union
import PIL

class VirtualImageEncoder(nn.Module):
    def __init__(self, output_dim=768, device="cuda"):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, output_dim)
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

    def forward(self, image_input: Union[str, PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Accepts image path (str), PIL.Image, or torch.Tensor of shape (3, H, W) or (B, 3, H, W).
        Returns: (B, output_dim)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

        elif isinstance(image_input, PIL.Image.Image):
            tensor = self.transform(image_input).unsqueeze(0).to(self.device)

        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                tensor = image_input.unsqueeze(0)
            elif image_input.dim() == 4:
                tensor = image_input
            else:
                raise ValueError("Tensor must be 3D or 4D (C, H, W) or (B, C, H, W)")
            tensor = tensor.to(self.device)

        else:
            raise ValueError("Unsupported input type for virtual image encoder.")

        with torch.no_grad():
            embedding = self.encoder(tensor)

        return embedding  # shape: (B, output_dim)



if __name__ == "__main__":
    image_path = "../../virtual_images/default.png"
    encoder = VirtualImageEncoder(output_dim=768, device="cuda")
    virtual_image = Image.open(image_path)

    embedding = encoder(virtual_image)
    print(f"Embedding shape: {embedding.shape}") # Embedding shape: torch.Size([1, 768])