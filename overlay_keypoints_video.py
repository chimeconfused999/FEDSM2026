import os
import cv2
import numpy as np
from valve_motion_analysis import classify_keypoints, match_points


def overlay_keypoints_on_video(
    video_path,
    masks_dir,
    output_path="valve_keypoints_overlay.avi",
    max_jump=20,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    prev = {k: None for k in ["tip_left", "tip_right", "base_left", "base_right"]}
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < len(mask_files):
            mask_path = os.path.join(masks_dir, mask_files[frame_index])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                binary_mask = (mask > 0).astype(np.uint8)
                keypoints = classify_keypoints(binary_mask)
            else:
                keypoints = None
        else:
            keypoints = None

        if keypoints is None:
            keypoints = {k: None for k in prev.keys()}

        tracked = {}
        for k in prev.keys():
            tracked[k] = match_points(prev[k], keypoints.get(k), max_jump=max_jump)
        prev = tracked

        # Draw overlay
        for name, color in [
            ("tip_left", (0, 0, 255)),
            ("tip_right", (255, 0, 0)),
            ("base_left", (0, 255, 255)),
            ("base_right", (255, 255, 0)),
        ]:
            pt = tracked.get(name)
            if pt is not None:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, -1)

        if tracked["base_left"] is not None and tracked["base_right"] is not None:
            bl = tuple(map(int, tracked["base_left"]))
            br = tuple(map(int, tracked["base_right"]))
            cv2.line(frame, bl, br, (0, 255, 0), 2)

            base_mid = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)
            if tracked["tip_left"] is not None:
                tl = tuple(map(int, tracked["tip_left"]))
                cv2.line(frame, base_mid, tl, (0, 0, 255), 2)
            if tracked["tip_right"] is not None:
                tr = tuple(map(int, tracked["tip_right"]))
                cv2.line(frame, base_mid, tr, (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Frame {frame_index}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"✅ Saved overlay video: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Overlay tracked valve keypoints on video.")
    parser.add_argument("--video", default="Ultrasound_Venous_Valve.avi", help="Input video path")
    parser.add_argument("--masks", default="predicted_masks", help="Directory with predicted masks")
    parser.add_argument("--out", default="valve_keypoints_overlay.avi", help="Output video path")
    parser.add_argument("--max-jump", type=float, default=20.0, help="Max pixel jump per frame")
    args = parser.parse_args()

    overlay_keypoints_on_video(
        video_path=args.video,
        masks_dir=args.masks,
        output_path=args.out,
        max_jump=args.max_jump,
    )
