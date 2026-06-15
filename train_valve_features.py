import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from model import UNetEdgeDetector
from dataset_safety import (
    VENOUS_IMAGE_DIR,
    VENOUS_MASK_DIR,
    VENOUS_MODEL,
    VENOUS_HISTORY,
    assert_training_config,
)
import cv2
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # Fallback if tqdm is not available
    HAS_TQDM = False
    def tqdm(iterable, desc="", **kwargs):
        return iterable

# ========== Default configuration (venous) ==========
IMAGE_DIR = VENOUS_IMAGE_DIR
MASK_DIR = VENOUS_MASK_DIR
MODEL_SAVE_PATH = VENOUS_MODEL
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2  # 20% for validation
SAVE_BEST_MODEL = True  # Save model with best validation loss

# Data augmentation parameters
USE_AUGMENTATION = True
AUGMENT_PROB = 0.5  # Probability of applying augmentation


def run_training(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    model_save_path=MODEL_SAVE_PATH,
    history_path=VENOUS_HISTORY,
    dataset="venous",
    pretrain_path=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    validation_split=VALIDATION_SPLIT,
    img_size=IMG_SIZE,
    use_augmentation=USE_AUGMENTATION,
    save_best_model=SAVE_BEST_MODEL,
):
    """Train U-Net with safety checks. Only writes model weights and history plot."""
    global IMG_SIZE, USE_AUGMENTATION
    IMG_SIZE = img_size
    USE_AUGMENTATION = use_augmentation

    assert_training_config(image_dir, mask_dir, model_save_path, history_path, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Image dir: {image_dir} (read-only)")
    print(f"Mask dir: {mask_dir} (read-only)")
    print(f"Model out: {model_save_path}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    print("\nLoading dataset...")
    full_dataset = ValveFeatureDataset(image_dir, mask_dir, augment=True)
    print(f"Found {len(full_dataset)} image-mask pairs")

    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_dataset.dataset.augment = False

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nInitializing model...")
    model = UNetEdgeDetector().to(device)
    if pretrain_path:
        if not os.path.isfile(pretrain_path):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrain_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = DiceBCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print("\nStarting training...\n")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    current_lr = learning_rate

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}")

        if (epoch + 1) % 5 == 0:
            metrics = calculate_metrics(model, val_loader, device)
            print(f"\nEpoch {epoch + 1} Metrics:")
            print(f"   IoU: {metrics['iou']:.4f}")
            print(f"   Dice: {metrics['dice']:.4f}")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")

        if save_best_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved (val_loss: {val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

    if not save_best_model:
        torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining complete! Model saved to {model_save_path}")
    plot_training_history(train_losses, val_losses, save_path=history_path)

    print("\nFinal Validation Metrics:")
    final_metrics = calculate_metrics(model, val_loader, device)
    for metric, value in final_metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")


# ========== Enhanced Dataset with Augmentation ==========
class ValveFeatureDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        
        # Find matching image-mask pairs
        self.filenames = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg')) and
               os.path.isfile(os.path.join(mask_dir, f))
        ]
        
        if len(self.filenames) == 0:
            raise ValueError(f"No matching image-mask pairs found in {image_dir} and {mask_dir}")
        
        self.base_transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor()
        ])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation if enabled
        if self.augment and USE_AUGMENTATION and np.random.rand() < AUGMENT_PROB:
            image, mask = self.apply_augmentation(image, mask)

        # Apply base transforms
        image = self.base_transform(image)
        
        # Transform mask separately to ensure it stays binary
        mask = T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()  # Ensure binary

        return image, mask

    def apply_augmentation(self, image, mask):
        """Apply random augmentations to both image and mask"""
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation (small angles to preserve valve structure)
        angle = np.random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
        
        # Random brightness/contrast (only on image, not mask)
        if np.random.rand() < 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if np.random.rand() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, mask


# ========== Enhanced Loss Function ==========
class DiceBCELoss(nn.Module):
    """Combined Dice Loss and BCE Loss for better segmentation"""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Combined loss
        return bce_loss + dice_loss


# ========== Training Function ==========
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        preds = model(images)
        loss = criterion(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        if HAS_TQDM:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        elif num_batches % 50 == 0:
            print(f"  Batch {num_batches}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / num_batches
    return avg_loss


# ========== Validation Function ==========
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            running_loss += loss.item()
            num_batches += 1
            if HAS_TQDM:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / num_batches
    return avg_loss


# ========== Calculate Metrics ==========
def calculate_metrics(model, dataloader, device, threshold=0.5):
    """Calculate IoU, Dice coefficient, and accuracy"""
    model.eval()
    iou_scores = []
    dice_scores = []
    accuracies = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            preds = torch.sigmoid(preds)
            preds_binary = (preds > threshold).float()
            
            # Flatten for metric calculation
            preds_flat = preds_binary.view(-1)
            masks_flat = masks.view(-1)
            
            # Intersection and Union
            intersection = (preds_flat * masks_flat).sum()
            union = preds_flat.sum() + masks_flat.sum() - intersection
            
            # IoU
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.append(iou.item())
            
            # Dice
            dice = (2 * intersection + 1e-6) / (preds_flat.sum() + masks_flat.sum() + 1e-6)
            dice_scores.append(dice.item())
            
            # Accuracy
            correct = (preds_flat == masks_flat).float().sum()
            accuracy = correct / len(preds_flat)
            accuracies.append(accuracy.item())
    
    return {
        'iou': np.mean(iou_scores),
        'dice': np.mean(dice_scores),
        'accuracy': np.mean(accuracies)
    }


# ========== Plot Training History ==========
def plot_training_history(train_losses, val_losses, save_path="training_history.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


# ========== Main Training Script ==========
if __name__ == "__main__":
    print("Tip: use train_venous.py or train_carotid.py for explicit, safer entry points.\n")
    run_training(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        model_save_path=MODEL_SAVE_PATH,
        history_path=VENOUS_HISTORY,
        dataset="venous",
    )
