import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from model import UNetEdgeDetector  # U-Net model must be defined in model.py

# --- Input Config ---
from pipeline_config import DEFAULT_MODEL, DEFAULT_VIDEO

video_path = DEFAULT_VIDEO
model_path = DEFAULT_MODEL

# --- Auto-increment output file ---
output_base = 'video'
output_ext = '.avi'
output_index = 1
while os.path.exists(f"{output_base}{output_index}{output_ext}"):
    output_index += 1
output_path = f"{output_base}{output_index}{output_ext}"

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetEdgeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Transform ---
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# --- Open Video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Resize and transform
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)

    # Get predicted edge map
    edge_map = torch.sigmoid(pred.squeeze()).cpu().numpy()

    # Apply strict thresholding to avoid blobs
    binary_mask = (edge_map > 0.5).astype(np.uint8) * 255
    resized_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Find contours only
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours in green (not filled)
    frame_with_edges = frame.copy()
    cv2.drawContours(frame_with_edges, contours, -1, (0, 255, 0), thickness=2)

    # Add frame number
    cv2.putText(frame_with_edges, f"Frame {frame_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame_with_edges)
    cv2.imshow("Green Edge Overlay", frame_with_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Saved new video as: {output_path}")
