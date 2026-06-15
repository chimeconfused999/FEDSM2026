import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from model import UNetEdgeDetector  # make sure this is defined in model.py

# --- Configuration ---
IMAGE_DIR = "images"
MASK_DIR = "masks_binary"
MODEL_SAVE_PATH = "trained_edge_model.pth"
IMG_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class EdgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.pairs = [f for f in os.listdir(image_dir)
                      if f in os.listdir(mask_dir) and f.endswith(('.jpg', '.png'))]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fname = self.pairs[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.img_transform(mask)
        mask = (mask > 0.5).float()

        # Optional: Dilate mask to slightly thicken edges
        mask_np = mask.squeeze().numpy()
        mask_np = cv2.dilate((mask_np * 255).astype(np.uint8),
                             np.ones((3, 3), np.uint8), iterations=1)
        mask = torch.tensor(mask_np / 255.0).unsqueeze(0).float()

        return image, mask

# --- Model Setup ---
model = UNetEdgeDetector().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE))

# --- Dataset & Loader ---
dataset = EdgeDataset(IMAGE_DIR, MASK_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📘 Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

# --- Save Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
