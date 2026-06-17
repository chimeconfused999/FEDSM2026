import os
import json
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from fedsm.model import UNetEdgeDetector
from fedsm.config import DEFAULT_MODEL, DEFAULT_VIDEO, DEFAULT_MASKS_DIR, DEFAULT_THRESHOLD, DEFAULT_IMG_SIZE
from fedsm.safety import assert_safe_output_dir, is_protected_venous_dir


def predict_masks(
    video_path,
    model_path,
    output_dir,
    threshold=0.5,
    img_size=(256, 256),
    confirm_overwrite_venous=False,
):
    if is_protected_venous_dir(output_dir):
        assert_safe_output_dir(
            output_dir,
            confirm_overwrite_venous=confirm_overwrite_venous,
            purpose="write predicted masks",
        )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEdgeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_index = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            pred = model(input_tensor)
            pred = torch.sigmoid(pred.squeeze()).cpu().numpy()
            binary_mask = (pred > threshold).astype("uint8") * 255

            mask_resized = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            out_path = os.path.join(output_dir, f"frame_{frame_index:04d}.png")
            cv2.imwrite(out_path, mask_resized)

            frame_index += 1

    cap.release()

    meta = {
        "video_path": video_path,
        "model_path": model_path,
        "fps": fps,
        "frame_count": frame_index,
        "frame_width": width,
        "frame_height": height,
        "threshold": threshold,
        "img_size": list(img_size),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {frame_index} masks to '{output_dir}'")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict binary masks for every video frame.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model weights")
    parser.add_argument("--out", default=DEFAULT_MASKS_DIR, help="Output directory for masks")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for binarization")
    parser.add_argument("--img-size", type=int, nargs=2, default=list(DEFAULT_IMG_SIZE), help="Model input size (w h)")
    parser.add_argument(
        "--confirm-overwrite-venous",
        action="store_true",
        help="Required when writing into predicted_masks/ or other protected venous folders",
    )
    args = parser.parse_args()

    predict_masks(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.out,
        threshold=args.threshold,
        img_size=(args.img_size[0], args.img_size[1]),
        confirm_overwrite_venous=args.confirm_overwrite_venous,
    )
