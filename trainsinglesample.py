import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNetEdgeDetector

# --- Config ---
IMAGE_PATH = "images"
MASK_PATH = "masks_binary"
IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10000
LR = 1e-4

# --- Load Image and Mask ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("L")

image = transform(image).unsqueeze(0).to(DEVICE)
mask = transform(mask)
mask = (mask > 0.5).float().unsqueeze(0).to(DEVICE)

# --- Model ---
model = UNetEdgeDetector().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Train ---
losses = []
for epoch in range(EPOCHS):
    model.train()
    pred = model(image)
    loss = criterion(pred, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

# --- Save model
torch.save(model.state_dict(), "trained_edge_model.pth")
print("✅ Model saved to trained_edge_model.pth")

# --- Evaluate
model.eval()
with torch.no_grad():
    pred = torch.sigmoid(model(image))[0, 0].cpu().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image[0].permute(1, 2, 0).cpu())
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask[0, 0].cpu(), cmap='gray')
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap='gray')
plt.title("Prediction")

plt.tight_layout()
plt.show()
