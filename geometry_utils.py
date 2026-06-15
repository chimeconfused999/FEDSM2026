"""Shared geometry extraction from binary valve segmentation masks."""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def load_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def mask_from_array(arr):
    return (arr > 0).astype(np.uint8)


def find_skeleton_endpoints(binary_mask):
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    h, w = skel.shape
    endpoints = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            neighborhood = skel[y - 1 : y + 2, x - 1 : x + 2]
            count = int(neighborhood.sum()) - 1
            if count == 1:
                endpoints.append((x, y))
    return skel, endpoints


def classify_keypoints(binary_mask):
    skel, endpoints = find_skeleton_endpoints(binary_mask)
    if len(endpoints) < 2:
        return None

    coords = np.array(endpoints, dtype=np.float32)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    if len(endpoints) >= 4:
        tips_idx = np.argsort(distances)[-2:]
        base_idx = np.argsort(distances)[:2]
        tips = coords[tips_idx]
        bases = coords[base_idx]
    else:
        tips_idx = np.argsort(distances)[-2:]
        tips = coords[tips_idx]
        base_candidates = coords[np.argsort(distances)[:2]]
        bases = base_candidates

    tip_left, tip_right = tips[np.argsort(tips[:, 0])]
    base_left, base_right = bases[np.argsort(bases[:, 0])]

    return {
        "tip_left": tuple(tip_left),
        "tip_right": tuple(tip_right),
        "base_left": tuple(base_left),
        "base_right": tuple(base_right),
    }


def compute_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def compute_lumen_area(binary_mask):
    """Cross-sectional lumen area proxy: foreground pixel count."""
    return float(np.sum(binary_mask > 0))


def compute_sinus_height(binary_mask):
    """
    Maximum perpendicular distance from vertical lumen centerline to outer boundary.
    Uses image column-wise extent as a 2D ultrasound cross-section proxy.
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return np.nan
    cx = (xs.min() + xs.max()) / 2.0
    left = cx - xs.min()
    right = xs.max() - cx
    return float(max(left, right))


def mean_contour_curvature(binary_mask):
    """Mean absolute curvature along the largest external contour."""
    contours, _ = cv2.findContours(
        (binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.nan, np.nan
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 5:
        return np.nan, np.nan

    pts = contour.astype(np.float64)
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-8
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return float(np.mean(curvature)), float(np.max(curvature))


def extract_frame_geometry(binary_mask):
    """Extract all geometric metrics for a single binary mask."""
    keypoints = classify_keypoints(binary_mask)
    angle = np.nan
    left_len = np.nan
    right_len = np.nan
    asymmetry = np.nan

    if keypoints is not None:
        base_mid = (np.array(keypoints["base_left"]) + np.array(keypoints["base_right"])) / 2.0
        v_left = np.array(keypoints["tip_left"]) - base_mid
        v_right = np.array(keypoints["tip_right"]) - base_mid
        angle = compute_angle(v_left, v_right)
        left_len = float(np.linalg.norm(v_left))
        right_len = float(np.linalg.norm(v_right))
        mean_len = (left_len + right_len) / 2.0
        if mean_len > 0:
            asymmetry = abs(left_len - right_len) / mean_len

    mean_curv, max_curv = mean_contour_curvature(binary_mask)
    valve_length = np.nanmean([left_len, right_len]) if np.isfinite(left_len) else np.nan

    return {
        "opening_angle_deg": angle,
        "leaflet_left_len_px": left_len,
        "leaflet_right_len_px": right_len,
        "valve_length_px": valve_length,
        "asymmetry": asymmetry,
        "lumen_area_px": compute_lumen_area(binary_mask),
        "sinus_height_px": compute_sinus_height(binary_mask),
        "leaflet_curvature_mean": mean_curv,
        "leaflet_curvature_max": max_curv,
    }


def dice_coefficient(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)


def iou_coefficient(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + 1e-6) / (union + 1e-6)


def pixel_accuracy(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    return float(np.mean(pred == gt))
