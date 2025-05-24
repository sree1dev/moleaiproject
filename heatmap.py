import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

# Dataset for Unlabelled Images
class MoleSSLDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        if self.transform:
            img = self.transform(img_pil)
            if img.ndim == 4:
                img = img.squeeze(0)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Unexpected tensor shape {img.shape} for {img_path}")
        return img, str(img_path)

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model.eval()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_image):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"Invalid input shape {input_image.shape}")
        input_image = input_image.cuda()
        self.model.zero_grad()
        output = self.model(input_image)
        target = output.max(dim=1)[0]  # Max activation in projection head
        score = target.item()  # Confidence score
        target.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        activations = self.activations
        heatmap = torch.mean(activations * pooled_gradients, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy(), score

def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlay, heatmap

# GUI for Image Selection
class HeatmapGUI:
    def __init__(self, root, model, grad_cam, transform, data_dir):
        self.root = root
        self.model = model
        self.grad_cam = grad_cam
        self.transform = transform
        self.data_dir = data_dir
        self.root.title("Grad-CAM Heatmap Generator")
        self.root.geometry("400x200")

        tk.Button(self.root, text="Select Images", command=self.select_images).pack(pady=10)
        tk.Button(self.root, text="Generate Heatmaps", command=self.generate_heatmaps).pack(pady=10)
        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=10)
        self.image_paths = []

    def select_images(self):
        self.image_paths = filedialog.askopenfilenames(
            initialdir=self.data_dir,
            title="Select Images",
            filetypes=[("PNG files", "*.png")]
        )
        if self.image_paths:
            messagebox.showinfo("Selected", f"Selected {len(self.image_paths)} images")
        else:
            messagebox.showwarning("No Selection", "No images selected")

    def generate_heatmaps(self):
        if not self.image_paths:
            messagebox.showerror("Error", "Please select images first")
            return
        dataset = MoleSSLDataset(self.image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)  # Batch for GPU efficiency

        for i, (images, img_paths) in enumerate(tqdm(dataloader, desc="Generating Heatmaps")):
            images = images.cuda()
            for j in range(images.shape[0]):
                try:
                    heatmap, score = self.grad_cam.generate(images[j])
                    original_img = cv2.imread(img_paths[j])
                    if original_img is None:
                        print(f"Failed to load original image: {img_paths[j]}")
                        continue
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    overlay, heatmap_raw = overlay_heatmap(heatmap, original_img)

                    # Create Enhanced Visualization
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_img)
                    plt.title(f"Original Image\n{Path(img_paths[j]).name}")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(original_img)
                    plt.imshow(heatmap_raw, cmap='jet', alpha=0.5)
                    plt.colorbar(label='Importance')
                    plt.title(f"Grad-CAM Heatmap\nConfidence: {score:.4f}")
                    plt.axis('off')
                    output_path = f"ssl_gradcam_{i*4+j+1}.png"
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close()
                    print(f"Saved Grad-CAM for {img_paths[j]} as {output_path}")
                except Exception as e:
                    print(f"Error processing {img_paths[j]}: {str(e)}")
        messagebox.showinfo("Complete", "Heatmap generation complete")

# Main Execution
def main():
    data_dir = r"C:\Users\anuse\Desktop\moleaiproject\Unlabelled"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Transform for Inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Pretrained SSL Model
    model = models.efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 128)  # Projection head
    )
    try:
        model.load_state_dict(torch.load("ssl_model.pth"))
    except FileNotFoundError:
        raise FileNotFoundError("Could not find ssl_model.pth")
    model = model.cuda() if torch.cuda.is_available() else model

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, model.features[-1])

    # Start GUI
    root = tk.Tk()
    app = HeatmapGUI(root, model, grad_cam, transform, data_dir)
    root.mainloop()

if __name__ == "__main__":
    main()