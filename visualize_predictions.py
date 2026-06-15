import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

# -----------------------
# Load your U-Net model
# -----------------------
class UNetEdgeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(2)

        self.down2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(2)

        self.bottom = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU()
        )

        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()
        )

        self.up2 = torch.nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU()
        )

        self.out = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b = self.bottom(p2)
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        return self.out(u2)

# -----------------------
# Dataset (no training)
# -----------------------
class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f in os.listdir(mask_dir) and f.endswith(('.png', '.jpg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, filename)).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, filename

# -----------------------
# Visualization
# -----------------------
def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    count = 0
    with torch.no_grad():
        for images, masks, filenames in dataloader:
            images = images.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds)

            for i in range(images.size(0)):
                input_img = images[i].permute(1, 2, 0).cpu().numpy()
                gt_mask = masks[i][0].cpu().numpy()
                pred_mask = preds[i][0].cpu().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(input_img)
                ax[0].set_title("Input Image")
                ax[1].imshow(gt_mask, cmap='gray')
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred_mask > 0.5, cmap='Greens')
                ax[2].set_title("Predicted Edges")
                for a in ax: a.axis('off')
                plt.suptitle(f"{filenames[i]}")
                plt.tight_layout()
                plt.show()

                count += 1
                if count >= num_images:
                    return

# -----------------------
# Run Visualization
# -----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = EdgeDataset("images", "masks_binary", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = UNetEdgeDetector().to(device)
    from pipeline_config import DEFAULT_MODEL

    model.load_state_dict(torch.load(DEFAULT_MODEL, map_location=device))

    visualize_predictions(model, dataloader, device, num_images=5)
