"""
Training script for valve keypoint detection (tips and bases).
This requires keypoint annotations in JSON format.

Keypoint annotation format (JSON):
{
    "image_name.jpg": {
        "tip_left": [x, y],
        "tip_right": [x, y],
        "base_left": [x, y],
        "base_right": [x, y]
    },
    ...
}
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from model import UNetEdgeDetector
import cv2
from tqdm import tqdm

# ========== Configuration ==========
IMAGE_DIR = "images"
KEYPOINT_ANNOTATIONS = "valve_keypoints.json"  # JSON file with keypoint annotations
MODEL_SAVE_PATH = "trained_keypoint_model.pth"
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEATMAP_SIGMA = 2.0  # Gaussian sigma for keypoint heatmaps

print(f"🔧 Using device: {DEVICE}")


# ========== Keypoint Dataset ==========
class KeypointDataset(Dataset):
    def __init__(self, image_dir, keypoint_file, img_size=(256, 256)):
        self.image_dir = image_dir
        self.img_size = img_size
        
        # Load keypoint annotations
        if not os.path.exists(keypoint_file):
            raise FileNotFoundError(
                f"Keypoint annotation file not found: {keypoint_file}\n"
                "Please create a JSON file with keypoint annotations."
            )
        
        with open(keypoint_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter to only include images that exist
        self.filenames = [
            fname for fname in self.annotations.keys()
            if os.path.exists(os.path.join(image_dir, fname))
        ]
        
        if len(self.filenames) == 0:
            raise ValueError("No valid image-keypoint pairs found!")
        
        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])
        
        print(f"✅ Loaded {len(self.filenames)} keypoint annotations")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (width, height)
        image = self.transform(image)
        
        # Get keypoints
        keypoints = self.annotations[img_name]
        
        # Scale keypoints to model input size
        scale_x = self.img_size[0] / orig_size[0]
        scale_y = self.img_size[1] / orig_size[1]
        
        scaled_keypoints = {}
        for kp_name, kp in keypoints.items():
            if kp is not None and len(kp) == 2:
                scaled_keypoints[kp_name] = [
                    kp[0] * scale_x,
                    kp[1] * scale_y
                ]
            else:
                scaled_keypoints[kp_name] = None
        
        # Create heatmaps for each keypoint
        heatmaps = self.create_heatmaps(scaled_keypoints)
        
        return image, heatmaps, scaled_keypoints

    def create_heatmaps(self, keypoints):
        """Create Gaussian heatmaps for keypoints"""
        h, w = self.img_size[1], self.img_size[0]
        heatmaps = np.zeros((4, h, w), dtype=np.float32)  # 4 keypoints
        
        keypoint_names = ['tip_left', 'tip_right', 'base_left', 'base_right']
        
        for i, kp_name in enumerate(keypoint_names):
            kp = keypoints.get(kp_name)
            if kp is not None:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < w and 0 <= y < h:
                    # Create Gaussian heatmap
                    y_coords, x_coords = np.ogrid[:h, :w]
                    heatmap = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * HEATMAP_SIGMA**2))
                    heatmaps[i] = heatmap
        
        return torch.tensor(heatmaps, dtype=torch.float32)


# ========== Keypoint Detection Model ==========
class KeypointDetectionModel(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            # Use UNet as backbone
            self.backbone = UNetEdgeDetector()
            # Modify output to predict 4 heatmaps (one per keypoint)
            self.backbone.out = nn.Conv2d(32, 4, kernel_size=1)
        else:
            self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ========== Loss Function ==========
class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_heatmaps, target_heatmaps):
        return self.mse(pred_heatmaps, target_heatmaps)


# ========== Training Functions ==========
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, heatmaps, _ in pbar:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        preds = model(images)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, heatmaps, _ in pbar:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss = criterion(preds, heatmaps)

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / len(dataloader)


def extract_keypoints_from_heatmaps(heatmaps, threshold=0.5):
    """Extract keypoint coordinates from predicted heatmaps"""
    keypoints = {}
    keypoint_names = ['tip_left', 'tip_right', 'base_left', 'base_right']
    
    heatmaps_np = heatmaps.cpu().numpy()
    
    for i, kp_name in enumerate(keypoint_names):
        heatmap = heatmaps_np[i]
        max_val = heatmap.max()
        
        if max_val > threshold:
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            keypoints[kp_name] = [float(x), float(y)]
        else:
            keypoints[kp_name] = None
    
    return keypoints


# ========== Main ==========
if __name__ == "__main__":
    print("\n📂 Loading keypoint dataset...")
    try:
        full_dataset = KeypointDataset(IMAGE_DIR, KEYPOINT_ANNOTATIONS, IMG_SIZE)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n💡 To create keypoint annotations, you can:")
        print("   1. Manually annotate images using a tool like LabelMe or CVAT")
        print("   2. Use the valve_motion_analysis.py script to extract keypoints from masks")
        print("   3. Create a JSON file with the format shown in the script header")
        exit(1)
    
    # Split dataset
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    print("\n🧠 Initializing model...")
    model = KeypointDetectionModel().to(DEVICE)
    
    # Loss and optimizer
    criterion = KeypointLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training
    print("\n🚀 Starting training...\n")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Best model saved (val_loss: {val_loss:.4f})")
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")
