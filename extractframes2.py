import cv2
import os

# --------- 1. Configuration ---------
video_path = 'video8.avi'
output_folder = 'images2'

# --------- 2. Create Output Folder ---------
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")
else:
    print(f"Saving frames to existing folder: {output_folder}")

# --------- 3. Load Video ---------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file '{video_path}'")
    exit()

# --------- 4. Frame Extraction ---------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✅ Saved: {filename}")
    frame_count += 1

cap.release()
print(f"\n🎉 Done! Extracted {frame_count} frames to '{output_folder}/'")
