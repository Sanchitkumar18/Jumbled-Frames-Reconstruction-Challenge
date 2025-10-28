import numpy as np
from typing import Dict as D
from typing import List as L
from tqdm import tqdm
import torch as t
import torchvision.models as models
import torchvision.transforms as Transforms
from PIL import Image as I
import os
import concurrent.futures

device = t.device("cuda" if t.cuda.is_available() else "cpu")#used to check if device has cuda-compatible gpu or not.
resnet = models.resnet18(pretrained=True)#this is used to create a CNN architecture and to load pretrained datasets.
resnet = t.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

transform = Transforms.Compose([
    Transforms.Resize((224, 224)),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def fetching_features(Path: str) -> np.ndarray:
    try:
        img = I.open(Path).convert("RGB")
        with t.no_grad():
            tensor = transform(img).unsqueeze(0).to(device)
            feature = resnet(tensor).squeeze().cpu().numpy()
        feature /= np.linalg.norm(feature) + 1e-8
        return feature
    except Exception as e:
        print(f"Feature extraction failed: {Path}: {e}")
        return np.zeros(512, dtype=np.float32)

def fetching_other_features(f_directory: str, f_name: L[str], max_workers: int = 4) -> D[str, np.ndarray]:
    features = {}
    paths = [os.path.join(f_directory, f) for f in f_name]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fname, feature in tqdm(zip(f_name, executor.map(fetching_features, paths)),
                                total=len(f_name), desc="other feature fetching..."):
            features[fname] = feature
    return features


def combining_features(classic_features: np.ndarray, other_features: np.ndarray) -> np.ndarray:
    normalization = classic_features / (np.linalg.norm(classic_features) + 1e-8)
    return np.concatenate([0.6 * normalization, 0.4 * other_features])


def cost_of_transition(A: np.ndarray, B: np.ndarray) -> float:
    difference = np.linalg.norm(A - B)
    brightness_difference = abs(np.mean(A[:16]) - np.mean(B[:16]))
    return difference + 0.4 * brightness_difference

def greedy_approach(features: D[str, np.ndarray], start_frame: str) -> L[str]:
    remaining = set(features.keys())
    order = [start_frame]
    remaining.remove(start_frame)
    while remaining:
        current = order[-1]
        best = min(remaining, key=lambda f: cost_of_transition(features[current], features[f]))
        order.append(best)
        remaining.remove(best)
    return order

def better_approach(order: L[str], features: D[str, np.ndarray]) -> L[str]:
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 2):
            a, b, c = order[i], order[i+1], order[i+2]
            cost1 = cost_of_transition(features[a], features[b]) + cost_of_transition(features[b], features[c])
            cost2 = cost_of_transition(features[a], features[c]) + cost_of_transition(features[c], features[b])
            if cost2 + 0.002 < cost1:
                order[i+1], order[i+2] = order[i+2], order[i+1]
                improved = True
    return order

def final_order_estimation(f_directory: str, features: D[str, np.ndarray]) -> L[str]:
    f_name = list(features.keys())

    other_features = fetching_other_features(f_directory, f_name)

    combined_features = {f: combining_features(features[f], other_features[f]) for f in f_name}

    worst = min(f_name, key=lambda f: np.mean(features[f][:16]))
    best = max(f_name, key=lambda f: np.mean(features[f][:16]))

    print("\nFrame order Estimation under progress...")
    forward = greedy_approach(combined_features, worst)
    backward = list(reversed(greedy_approach(combined_features, best)))

    def total_cost(order):
        return sum(cost_of_transition(combined_features[order[i]], combined_features[order[i+1]])
                   for i in range(len(order)-1))

    best = forward if total_cost(forward) <= total_cost(backward) else backward
    best = better_approach(best, combined_features)

    print(f"Estimated final order have ({len(best)} frames)")
    return best
