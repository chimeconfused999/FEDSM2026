import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# --------- Dataset Class ---------
class EdgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Only load filenames that have both image and mask
        self.filenames = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg')) and
               os.path.isfile(os.path.join(mask_dir, f))
        ]

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale binary mask

        return self.transform(image), self.transform(mask)

# --------- Simple CNN Model ---------
class SimpleEdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --------- Training Function ---------
def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# --------- Main Script ---------
if __name__ == "__main__":
    image_dir = "images"
    mask_dir = "masks_binary"  # generated from your drawing conversion step

    dataset = EdgeDataset(image_dir, mask_dir)

    if len(dataset) == 0:
        print("❌ No matched image-mask pairs found. Add more masks and try again.")
        exit()

    print(f"✅ Found {len(dataset)} image-mask pairs for training.")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEdgeNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train(model, dataloader, criterion, optimizer, device, epochs=10)

    # Save model
    torch.save(model.state_dict(), "trained_edge_model.pth")
    print("✅ Training complete. Model saved as trained_edge_model.pth")
