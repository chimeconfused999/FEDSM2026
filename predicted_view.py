import cv2
import torch
import torchvision.transforms as T
from model import UNetEdgeDetector
from PIL import Image
import numpy as np

# --- Config ---
from pipeline_config import DEFAULT_MODEL, DEFAULT_VIDEO

video_path = DEFAULT_VIDEO
model_path = DEFAULT_MODEL
img_size = (256, 256)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetEdgeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Transform ---
transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])

# --- Load Video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video.")
    exit()

# --- Inference Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL and preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    resized = pil_img.resize(img_size)
    input_tensor = transform(resized).unsqueeze(0).to(device)

    # Predict edges
    with torch.no_grad():
        pred = model(input_tensor)

    pred = torch.sigmoid(pred.squeeze()).cpu().numpy()
    binary_mask = (pred > 0.57).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Show prediction
    cv2.imshow("Predicted Edge View", binary_mask_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
