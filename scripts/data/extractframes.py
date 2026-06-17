import argparse
import cv2
import os

from cinevalve.safety import assert_safe_output_dir
from cinevalve.config import DEFAULT_VIDEO


def extract_frames(video_path, output_folder, confirm_overwrite_venous=False):
    assert_safe_output_dir(
        output_folder,
        confirm_overwrite_venous=confirm_overwrite_venous,
        purpose="extract frames",
    )

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file '{video_path}'")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        if frame_count % 100 == 0:
            print(f"Saved: {filename}")
        frame_count += 1

    cap.release()
    print(f"Done. Extracted {frame_count} frames to '{output_folder}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from ultrasound video.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path")
    parser.add_argument("--out", default="images", help="Output folder for frames")
    parser.add_argument(
        "--confirm-overwrite-venous",
        action="store_true",
        help="Required when writing into a protected venous folder (images/, etc.)",
    )
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.confirm_overwrite_venous)
