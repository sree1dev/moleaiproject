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
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import torch.nn.functional as F

# Transform matching training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple dataset class
class MoleDataset(Dataset):
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
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.tensor(np.array(img_pil).transpose(2, 0, 1), dtype=torch.float32) / 255.0
            
        return img_tensor, str(img_path)

# Exact same GradCAM class from training code
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.requires_grad_(True)
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[:, target_class].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.detach().cpu().numpy()

    def cleanup(self):
        for handle in self.hook_handles:
            handle.remove()

# Function to create heatmap visualization
def create_heatmap_visualization(heatmap, original_image, image_name, score):
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Create colored heatmap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    # Create overlay
    overlay = 0.6 * original_image + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title(f'Original\n{image_name}')
    ax1.axis('off')
    
    # Heatmap only
    im2 = ax2.imshow(heatmap_resized, cmap='jet')
    ax2.set_title(f'Activation Map\nScore: {score:.4f}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

# GUI Application
class GradCAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mole Grad-CAM Analyzer")
        self.root.geometry("500x300")
        
        # Variables
        self.model = None
        self.grad_cam = None
        self.selected_images = []
        self.data_dir = r"C:\Users\anuse\Desktop\moleaiproject\Unlabelled"
        
        # Load model on startup
        self.load_model()
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Mole Grad-CAM Analysis", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Model status
        if self.model is not None:
            status_text = "✓ Model loaded successfully"
            status_color = "green"
        else:
            status_text = "✗ Failed to load model"
            status_color = "red"
            
        status_label = tk.Label(self.root, text=status_text, 
                               fg=status_color, font=("Arial", 12))
        status_label.pack(pady=10)
        
        # Buttons
        select_btn = tk.Button(self.root, text="Select Images", 
                              command=self.select_images,
                              bg="lightblue", font=("Arial", 12),
                              width=20, height=2)
        select_btn.pack(pady=10)
        
        generate_btn = tk.Button(self.root, text="Generate Heatmaps", 
                                command=self.generate_heatmaps,
                                bg="lightgreen", font=("Arial", 12),
                                width=20, height=2)
        generate_btn.pack(pady=10)
        
        exit_btn = tk.Button(self.root, text="Exit", 
                            command=self.root.quit,
                            bg="lightcoral", font=("Arial", 12),
                            width=20, height=2)
        exit_btn.pack(pady=10)
        
        # Info label
        self.info_label = tk.Label(self.root, text="No images selected", 
                                  font=("Arial", 10))
        self.info_label.pack(pady=10)
        
    def load_model(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            # Create model with same architecture as training
            self.model = models.efficientnet_b0()
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[1].in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            
            # Load trained weights
            model_path = "ssl_model_augment_tuned.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model = self.model.to(device)
            
            # Initialize GradCAM
            self.grad_cam = SimpleGradCAM(self.model, self.model.features[-1])
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.grad_cam = None
            
    def select_images(self):
        if not os.path.exists(self.data_dir):
            messagebox.showerror("Error", f"Data directory not found: {self.data_dir}")
            return
            
        self.selected_images = filedialog.askopenfilenames(
            initialdir=self.data_dir,
            title="Select Mole Images",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if self.selected_images:
            self.info_label.config(text=f"Selected {len(self.selected_images)} images")
            messagebox.showinfo("Success", f"Selected {len(self.selected_images)} images")
        else:
            self.info_label.config(text="No images selected")
            
    def generate_heatmaps(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        if not self.selected_images:
            messagebox.showerror("Error", "Please select images first!")
            return
            
        # Create output directory
        output_dir = os.path.join(os.path.dirname(self.data_dir), "gradcam_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images
        successful_count = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i, img_path in enumerate(tqdm(self.selected_images, desc="Processing images")):
            try:
                # Load and preprocess image
                original_img = cv2.imread(img_path)
                if original_img is None:
                    print(f"Failed to load: {img_path}")
                    continue
                    
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                original_img_normalized = original_img.astype(np.float32) / 255.0
                
                # Prepare tensor
                img_pil = Image.fromarray(original_img)
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                # Generate heatmap
                cam = self.grad_cam.generate(img_tensor)
                heatmap = cam[0, 0]  # Extract single heatmap
                
                # Get activation score
                with torch.no_grad():
                    output = self.model(img_tensor)
                    target_class = output.argmax(dim=1).item()
                    score = output[0, target_class].item()
                
                # Create visualization
                image_name = os.path.basename(img_path)
                fig = create_heatmap_visualization(heatmap, original_img_normalized, 
                                                 image_name, score)
                
                # Save result
                output_path = os.path.join(output_dir, f"gradcam_{i+1:03d}_{image_name}")
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                successful_count += 1
                print(f"Processed: {image_name} -> Score: {score:.4f}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Cleanup
        if self.grad_cam:
            self.grad_cam.cleanup()
            
        # Show results
        messagebox.showinfo("Complete", 
                           f"Processing complete!\n"
                           f"Successfully processed: {successful_count}/{len(self.selected_images)} images\n"
                           f"Results saved to: {output_dir}")

def main():
    # Check if data directory exists
    data_dir = r"C:\Users\anuse\Desktop\moleaiproject\Unlabelled"
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        
    # Check if model file exists
    model_path = "ssl_model_augment_tuned.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please make sure the trained model file is in the current directory.")
        return
        
    # Create and run GUI
    root = tk.Tk()
    app = GradCAMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()