import cv2
import numpy as np
import os
from typing import Dict

def compute_features(frame_path: str) -> np.ndarray:
    """
    Computes a refined feature vector for a frame.
    Combines:
        - Brightness grid (4x4 = 16)
        - Edge intensity (1)
        - Mean hue (1)
        - Gradient orientation histogram (8 bins)
    → Total length = 26-D vector
    """
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Brightness normalization (stabilize exposure) ---
    gray = cv2.equalizeHist(gray)

    # --- Brightness grid (4×4) ---
    h, w = gray.shape
    grid = []
    for i in range(4):
        for j in range(4):
            patch = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            grid.append(np.mean(patch) / 255.0)
    grid = np.array(grid)

    # --- Edge magnitude (Sobel) ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    edge_strength = np.mean(mag) / 255.0

    # --- Gradient orientation histogram (8 bins) ---
    angle = np.arctan2(sobely, sobelx)
    angle = np.degrees(angle) % 180
    hist, _ = np.histogram(angle, bins=8, range=(0, 180), weights=mag)
    hist = hist / (np.sum(hist) + 1e-6)  # normalize

    # --- Mean hue (normalized) ---
    hue_mean = np.mean(hsv[:, :, 0]) / 180.0

    features = np.concatenate([grid, [edge_strength, hue_mean], hist])
    return features.astype(np.float32)

def load_features(frames_dir: str) -> Dict[str, np.ndarray]:
    """
    Loads and computes features for all frames in a directory.
    """
    features = {}
    for fname in sorted(os.listdir(frames_dir)):
        if fname.lower().endswith((".jpg", ".png")):
            path = os.path.join(frames_dir, fname)
            features[fname] = compute_features(path)
    return features
