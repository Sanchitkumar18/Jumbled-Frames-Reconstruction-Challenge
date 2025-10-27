import numpy as np
from typing import Dict, List
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import concurrent.futures

# ======= Initialize CNN model =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ======= AI embedding extraction =======
def get_ai_features(frame_path: str) -> np.ndarray:
    try:
        img = Image.open(frame_path).convert("RGB")
        with torch.no_grad():
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).squeeze().cpu().numpy()
        feat /= np.linalg.norm(feat) + 1e-8
        return feat
    except Exception as e:
        print(f"[Warning] Failed to extract AI features from {frame_path}: {e}")
        return np.zeros(512, dtype=np.float32)

def compute_ai_embeddings(frames_dir: str, frame_names: List[str], max_workers: int = 4) -> Dict[str, np.ndarray]:
    """Compute CNN features in parallel for speed."""
    ai_features = {}
    paths = [os.path.join(frames_dir, f) for f in frame_names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fname, feat in tqdm(zip(frame_names, executor.map(get_ai_features, paths)),
                                total=len(frame_names), desc="AI Embeddings"):
            ai_features[fname] = feat
    return ai_features

# ======= Hybrid feature fusion =======
def combine_features(classic_feat: np.ndarray, ai_feat: np.ndarray) -> np.ndarray:
    classic_norm = classic_feat / (np.linalg.norm(classic_feat) + 1e-8)
    return np.concatenate([0.6 * classic_norm, 0.4 * ai_feat])

# ======= Cost and sequencing =======
def transition_cost(v1: np.ndarray, v2: np.ndarray) -> float:
    diff = np.linalg.norm(v1 - v2)
    brightness_diff = abs(np.mean(v1[:16]) - np.mean(v2[:16]))
    return diff + 0.4 * brightness_diff

def greedy_sequence(features: Dict[str, np.ndarray], start_frame: str) -> List[str]:
    remaining = set(features.keys())
    order = [start_frame]
    remaining.remove(start_frame)
    while remaining:
        current = order[-1]
        best = min(remaining, key=lambda f: transition_cost(features[current], features[f]))
        order.append(best)
        remaining.remove(best)
    return order

def smooth_sequence(order: List[str], features: Dict[str, np.ndarray]) -> List[str]:
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 2):
            a, b, c = order[i], order[i+1], order[i+2]
            cost1 = transition_cost(features[a], features[b]) + transition_cost(features[b], features[c])
            cost2 = transition_cost(features[a], features[c]) + transition_cost(features[c], features[b])
            if cost2 + 0.002 < cost1:
                order[i+1], order[i+2] = order[i+2], order[i+1]
                improved = True
    return order

# ======= Main estimation function =======
def estimate_order(frames_dir: str, features: Dict[str, np.ndarray]) -> List[str]:
    frame_names = list(features.keys())

    # Compute AI embeddings
    ai_feats = compute_ai_embeddings(frames_dir, frame_names)

    # Fuse handcrafted + AI features
    fused_features = {f: combine_features(features[f], ai_feats[f]) for f in frame_names}

    # Start from extreme brightness frames
    darkest = min(frame_names, key=lambda f: np.mean(features[f][:16]))
    brightest = max(frame_names, key=lambda f: np.mean(features[f][:16]))

    print("\nEstimating frame order using Hybrid AI + Greedy...")
    forward = greedy_sequence(fused_features, darkest)
    backward = list(reversed(greedy_sequence(fused_features, brightest)))

    def total_cost(order):
        return sum(transition_cost(fused_features[order[i]], fused_features[order[i+1]])
                   for i in range(len(order)-1))

    best = forward if total_cost(forward) <= total_cost(backward) else backward
    best = smooth_sequence(best, fused_features)

    print(f"✅ Final order estimated ({len(best)} frames)")
    return best
