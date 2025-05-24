import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# Enhanced SimCLR Augmentation for SSL
ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Subtle rotation
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),  # Stronger jitter
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Light blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Dataset for SSL (no labels)
class MoleSSLDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert to PIL Image
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        return torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0, img

# SimCLR Loss
def simclr_loss(features1, features2, temperature=0.5):
    batch_size = features1.size(0)
    labels = torch.arange(batch_size).cuda()
    features = torch.cat([features1, features2], dim=0)
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool).cuda()
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    labels = torch.cat([labels, labels], dim=0)
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss

# Load Data
data_dir = r"C:\Users\anuse\Desktop\moleaiproject\Unlabelled"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory not found: {data_dir}")

image_paths = [Path(data_dir) / f for f in os.listdir(data_dir) if f.endswith(".png")]
print(f"Total images: {len(image_paths)}")
assert len(image_paths) == 1156, f"Expected 1156 images, got {len(image_paths)}"

# SSL Pretraining
ssl_dataset = MoleSSLDataset(image_paths, transform=ssl_transform)
ssl_loader = DataLoader(ssl_dataset, batch_size=16, shuffle=True)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 128)  # Projection head for SimCLR
)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Lower learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

for epoch in range(30):  # Increased to 30 epochs
    model.train()
    ssl_loss = 0
    for img1, img2 in tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}"):
        img1, img2 = img1.cuda(), img2.cuda()
        optimizer.zero_grad()
        features1 = model(img1)
        features2 = model(img2)
        loss = simclr_loss(features1, features2)
        loss.backward()
        optimizer.step()
        ssl_loss += loss.item()
    avg_loss = ssl_loss / len(ssl_loader)
    print(f"SSL Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    scheduler.step(avg_loss)
    if scheduler.get_last_lr()[0] < optimizer.param_groups[0]['lr']:
        print(f"Learning rate reduced to {scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), "ssl_model.pth")
print("SSL pretraining complete. Model saved as 'ssl_model.pth'.")

# Placeholder for Supervised Fine-Tuning
print("To proceed with fine-tuning, provide a CSV file with columns 'filename' and 'label' (0=benign, 1=malignant).")