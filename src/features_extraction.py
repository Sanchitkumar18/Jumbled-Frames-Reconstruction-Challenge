import cv2
import numpy as np
import os
from typing import Dict


def compute_features(frame_path: str) -> np.ndarray:
    """
    Computes a custom feature vector for a frame.

    Feature = [mean brightness grid(4×4) + edge intensity + hue mean]
    → Total length = 16 + 1 + 1 = 18-D vector
    """
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Brightness grid
    h, w = gray.shape
    grid = []
    for i in range(4):
        for j in range(4):
            patch = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            grid.append(np.mean(patch) / 255.0)
    grid = np.array(grid)

    # Edge magnitude (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0

    # Mean hue (normalized)
    hue_mean = np.mean(hsv[:, :, 0]) / 180.0

    features = np.concatenate([grid, [edge_strength, hue_mean]])
    return features


def load_features(frames_dir: str) -> Dict[str, np.ndarray]:
    """
    Loads and computes features for all frames in a directory.
    """
    features = {}
    for fname in sorted(os.listdir(frames_dir)):
        if fname.endswith(".jpg"):
            path = os.path.join(frames_dir, fname)
            features[fname] = compute_features(path)
    return features
