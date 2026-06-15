# Valve Feature Detection Training Guide

This guide explains how to train models to identify valve features in ultrasound images.

## Overview

There are two main approaches to training valve feature detection:

1. **Segmentation-based** (`train_valve_features.py`) - Trains a model to segment valve regions
2. **Keypoint-based** (`train_valve_keypoints.py`) - Trains a model to detect specific valve keypoints (tips, bases)

## Prerequisites

- Python 3.7+
- PyTorch
- Required packages: `torch`, `torchvision`, `PIL`, `numpy`, `opencv-python`, `matplotlib`, `tqdm`, `scikit-image`

Install dependencies:
```bash
pip install torch torchvision pillow numpy opencv-python matplotlib tqdm scikit-image
```

## Directory Structure

Ensure you have the following structure:
```
FEDSM 2026/
├── images/              # Input ultrasound images
├── masks_binary/        # Binary masks (ground truth segmentations)
├── model.py            # UNet model definition
├── train_valve_features.py
└── train_valve_keypoints.py
```

## Method 1: Segmentation-Based Training

This method trains the model to identify valve regions by learning to segment them.

### Step 1: Prepare Your Data

1. **Images**: Place your ultrasound images in the `images/` directory
2. **Masks**: Create corresponding binary masks in `masks_binary/` directory
   - Masks should be binary (black=background, white=valve)
   - Same filename as corresponding image
   - Format: PNG or JPG

### Step 2: Configure Training

Edit `train_valve_features.py` to adjust:
- `IMAGE_DIR`: Directory with input images
- `MASK_DIR`: Directory with binary masks
- `BATCH_SIZE`: Based on your GPU memory (default: 4)
- `EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

### Step 3: Run Training

```bash
python train_valve_features.py
```

### Step 4: Monitor Training

The script will:
- Display training and validation loss for each epoch
- Calculate metrics (IoU, Dice coefficient, Accuracy) every 5 epochs
- Save the best model based on validation loss
- Generate a training history plot

### Features Identified

This model learns to identify:
- **Valve boundaries**: Precise edges of valve structures
- **Valve regions**: Complete segmentation of valve areas
- **Valve shape**: Learns the characteristic shape patterns

## Method 2: Keypoint-Based Training

This method trains the model to detect specific valve keypoints (tips and bases).

### Step 1: Create Keypoint Annotations

Create a JSON file (`valve_keypoints.json`) with keypoint coordinates:

```json
{
    "image001.jpg": {
        "tip_left": [120, 150],
        "tip_right": [180, 145],
        "base_left": [110, 200],
        "base_right": [190, 195]
    },
    "image002.jpg": {
        "tip_left": [125, 155],
        "tip_right": [185, 150],
        "base_left": [115, 205],
        "base_right": [195, 200]
    }
}
```

### Step 2: Generate Annotations from Masks (Optional)

You can use the existing `valve_motion_analysis.py` script to extract keypoints from masks:

```python
from valve_motion_analysis import classify_keypoints, load_binary_mask
import json
import os

annotations = {}
mask_dir = "masks_binary"

for fname in os.listdir(mask_dir):
    if fname.endswith(('.png', '.jpg')):
        mask = load_binary_mask(os.path.join(mask_dir, fname))
        if mask is not None:
            keypoints = classify_keypoints(mask)
            if keypoints:
                annotations[fname] = keypoints

with open("valve_keypoints.json", "w") as f:
    json.dump(annotations, f, indent=2)
```

### Step 3: Run Training

```bash
python train_valve_keypoints.py
```

### Features Identified

This model learns to identify:
- **Tip positions**: Left and right valve leaflet tips
- **Base positions**: Left and right valve leaflet bases
- **Valve geometry**: Spatial relationships between keypoints

## Training Tips

### 1. Data Quality
- Ensure masks are accurate and consistent
- Remove low-quality images from training
- Balance your dataset (include various valve states: open, closed, intermediate)

### 2. Data Augmentation
The segmentation training script includes:
- Random horizontal flips
- Small rotations (±10°)
- Brightness/contrast adjustments

Adjust `AUGMENT_PROB` in `train_valve_features.py` to control augmentation frequency.

### 3. Hyperparameters

**Learning Rate**: Start with 1e-4, reduce if loss doesn't decrease
**Batch Size**: Increase if you have GPU memory (faster training)
**Epochs**: Monitor validation loss - stop if it plateaus

### 4. Model Architecture

The default model uses a UNet architecture (`model.py`). You can:
- Modify the UNet depth/channels for more capacity
- Use pretrained backbones (e.g., ResNet encoder)
- Add attention mechanisms for better feature focus

### 5. Loss Functions

The segmentation script uses **Dice + BCE Loss** which works well for imbalanced data (small valve regions vs large background).

For keypoint detection, **MSE Loss** on heatmaps is used.

## Using Trained Models

### Segmentation Model

```python
import torch
from model import UNetEdgeDetector
from PIL import Image
import torchvision.transforms as T

# Load model
model = UNetEdgeDetector()
model.load_state_dict(torch.load("trained_valve_model.pth"))
model.eval()

# Predict
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
image = Image.open("test_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    pred = model(input_tensor)
    mask = torch.sigmoid(pred) > 0.5
```

### Keypoint Model

```python
import torch
from train_valve_keypoints import KeypointDetectionModel, extract_keypoints_from_heatmaps

# Load model
model = KeypointDetectionModel()
model.load_state_dict(torch.load("trained_keypoint_model.pth"))
model.eval()

# Predict and extract keypoints
with torch.no_grad():
    pred_heatmaps = model(input_tensor)
    keypoints = extract_keypoints_from_heatmaps(pred_heatmaps[0])
```

## Troubleshooting

### Low Accuracy
- Check mask quality and consistency
- Increase training data
- Adjust learning rate
- Try different loss functions

### Overfitting
- Increase data augmentation
- Add dropout to model
- Reduce model capacity
- Use more training data

### Out of Memory
- Reduce batch size
- Reduce image size
- Use gradient accumulation

### Slow Training
- Use GPU if available
- Increase batch size (if memory allows)
- Reduce image resolution
- Use mixed precision training

## Next Steps

After training:
1. Evaluate on test set
2. Use `predict_masks.py` for inference on videos
3. Use `valve_motion_analysis.py` to analyze valve dynamics
4. Generate 3D models with `merge.py` or `3dmodel.py`

## Advanced: Multi-Task Learning

You can combine both approaches by training a model that predicts both segmentation masks and keypoints simultaneously. This would require:
- A shared encoder (UNet backbone)
- Two decoder heads (one for segmentation, one for keypoints)
- Combined loss function

This is more complex but can improve performance by learning complementary features.
