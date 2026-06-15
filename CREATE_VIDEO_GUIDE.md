# Guide: Creating Video with Geometrical Features

This guide shows you how to create a video with valve geometrical features overlaid.

## Option 1: Using Trained Model (Recommended)

If you have a trained model, generate masks first, then overlay features:

### Step 1: Generate Masks from Video
```bash
python predict_masks.py --video "Ultrasound_Venous_Valve.avi" --model "trained_valve_model.pth" --out "predicted_masks"
```

### Step 2: Create Video with Keypoints Overlay
```bash
python overlay_keypoints_video.py --video "Ultrasound_Venous_Valve.avi" --masks "predicted_masks" --out "valve_features_video.avi"
```

This will create a video showing:
- **Red circles**: Left tip
- **Blue circles**: Right tip  
- **Cyan circles**: Left base
- **Yellow circles**: Right base
- **Green line**: Base connection
- **Red/Blue lines**: Tip-to-base connections

---

## Option 2: Using Existing Binary Masks

If you already have binary masks, you can use them directly:

### Step 1: Create Video with Keypoints
```bash
python overlay_keypoints_video.py --video "Ultrasound_Venous_Valve.avi" --masks "masks_binary2" --out "valve_features_video.avi"
```

---

## Option 3: Edge/Contour Overlay

For a simpler edge overlay (green contours):

```bash
python annotatevideo.py
```

This uses:
- Video: `Ultrasound_Venous_Valve.avi`
- Model: `trained_edge_model.pth`
- Output: Auto-incremented `video1.avi`, `video2.avi`, etc.

---

## Option 4: Complete Workflow with Metrics

For a complete analysis with metrics:

### Step 1: Analyze Masks and Generate Metrics
```bash
python valve_motion_analysis.py --masks "predicted_masks" --fps 30.0 --out "valve_metrics"
```

This creates:
- `valve_metrics.csv` - Detailed metrics per frame
- `valve_metrics_plots.png` - Visualization plots
- `valve_metrics_summary.json` - Summary statistics

### Step 2: Create Video with Keypoints
```bash
python overlay_keypoints_video.py --video "Ultrasound_Venous_Valve.avi" --masks "predicted_masks" --out "valve_features_video.avi"
```

---

## Quick Command Reference

### Basic Keypoint Video (using existing masks):
```bash
python overlay_keypoints_video.py --video "Ultrasound_Venous_Valve.avi" --masks "masks_binary2"
```

### With Custom Output:
```bash
python overlay_keypoints_video.py --video "your_video.avi" --masks "your_masks" --out "output.avi"
```

### With Custom Max Jump (for tracking):
```bash
python overlay_keypoints_video.py --video "Ultrasound_Venous_Valve.avi" --masks "masks_binary2" --max-jump 30
```

---

## What Each Video Shows

### `overlay_keypoints_video.py` Output:
- **Geometrical features**: Tips, bases, connections
- **Tracking**: Smooth keypoint tracking across frames
- **Frame numbers**: Displayed on each frame

### `annotatevideo.py` Output:
- **Edge contours**: Green outlines of valve boundaries
- **Frame numbers**: Displayed on each frame

---

## Troubleshooting

### "Video not found" error:
- Check the video file path
- Use full path if needed: `--video "C:/path/to/video.avi"`

### "Masks directory not found" error:
- Ensure masks directory exists
- Check mask filenames match video frames

### Keypoints not showing:
- Masks may not have detectable keypoints
- Try adjusting `--max-jump` parameter
- Check mask quality

### Video codec issues:
- If video won't play, try changing codec in the script
- XVID works on most systems
- Alternative: Use MP4V codec
