import cv2
import numpy as np
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import Dict
from tqdm import tqdm

# ======== CNN setup (for optional AI feature fusion) ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ======== HANDCRAFTED FEATURES ========
def compute_features(frame_path: str) -> np.ndarray:
    """
    Computes a refined handcrafted feature vector for a frame.
    Components:
        - Brightness grid (4×4 = 16)
        - Edge intensity (1)
        - Mean hue (1)
        - Gradient orientation histogram (8 bins)
    → Total length = 26-D vector
    """
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Normalize brightness
    gray = cv2.equalizeHist(gray)

    # Brightness grid
    h, w = gray.shape
    grid = [np.mean(gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]) / 255.0
            for i in range(4) for j in range(4)]
    grid = np.array(grid)

    # Edge magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    edge_strength = np.mean(mag) / 255.0

    # Gradient orientation histogram
    angle = np.degrees(np.arctan2(sobely, sobelx)) % 180
    hist, _ = np.histogram(angle, bins=8, range=(0, 180), weights=mag)
    hist = hist / (np.sum(hist) + 1e-6)

    # Mean hue
    hue_mean = np.mean(hsv[:, :, 0]) / 180.0

    features = np.concatenate([grid, [edge_strength, hue_mean], hist])
    return features.astype(np.float32)


# ======== CNN (AI) FEATURES ========
def compute_ai_features(frame_path: str) -> np.ndarray:
    try:
        img = Image.open(frame_path).convert("RGB")
        with torch.no_grad():
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).squeeze().cpu().numpy()
        feat /= np.linalg.norm(feat) + 1e-8
        return feat.astype(np.float32)
    except Exception as e:
        print(f"[Warning] AI feature extraction failed for {frame_path}: {e}")
        return np.zeros(512, dtype=np.float32)


# ======== CACHE UTILITIES ========
def compute_and_cache_features(frames_dir: str, cache_path: str = "data/features_cache.pkl") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute (handcrafted + CNN) features for all frames and save to cache.
    Returns a dict: { frame_name: { "classic": ..., "ai": ... } }
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"🔁 Using cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    features = {}
    frame_files = sorted([f for f in os.listdir(frames_dir)
                          if f.lower().endswith((".jpg", ".png"))])

    print(f"🧮 Computing features for {len(frame_files)} frames...")
    for fname in tqdm(frame_files, desc="Feature Extraction"):
        path = os.path.join(frames_dir, fname)
        features[fname] = {
            "classic": compute_features(path),
            "ai": compute_ai_features(path)
        }

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    print(f"✅ Features cached at {cache_path}")
    return features


def load_cached_features(cache_path: str = "data/features_cache.pkl") -> Dict[str, Dict[str, np.ndarray]]:
    """Load precomputed features from cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)
