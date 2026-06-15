import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from model import UNetEdgeDetector  # Make sure this is defined in model.py

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pipeline_config import DEFAULT_MODEL, DEFAULT_VIDEO

model_path = DEFAULT_MODEL
video_path = DEFAULT_VIDEO

# Load model
model = UNetEdgeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Video setup
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('confidence_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        confidence_map = torch.sigmoid(output)[0, 0].cpu().numpy()  # shape: (256, 256)

    # Normalize to 0–255 and resize back to original video size
    confidence_norm = (confidence_map * 255).astype(np.uint8)
    confidence_resized = cv2.resize(confidence_norm, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply colormap to visualize confidence
    heatmap = cv2.applyColorMap(confidence_resized, cv2.COLORMAP_JET)

    # Optional: Overlay text
    cv2.putText(heatmap, "Confidence Map", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(heatmap)
    cv2.imshow("Confidence View", heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Confidence heatmap video saved as 'confidence_output.avi'")
