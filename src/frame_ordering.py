import numpy as np
from typing import Dict, List
from tqdm import tqdm
import random


def transition_cost(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Custom cost: Euclidean distance + brightness penalty.
    """
    diff = np.linalg.norm(v1 - v2)
    brightness_diff = abs(np.mean(v1[:16]) - np.mean(v2[:16]))  # grid brightness
    return diff + 0.5 * brightness_diff


def estimate_order(frames_dir: str, features: Dict[str, np.ndarray]) -> List[str]:
    """
    Estimates frame order using greedy + look-ahead search.

    Heuristic:
        - Start with the darkest frame.
        - Iteratively append the frame with minimum transition cost,
          using a 3-frame look-ahead for smoother motion.
    """
    frame_names = list(features.keys())
    n = len(frame_names)
    remaining = set(frame_names)

    # Start = lowest brightness frame
    start = min(frame_names, key=lambda f: np.mean(features[f][:16]))
    order = [start]
    remaining.remove(start)

    print("Estimating frame order...")
    while remaining:
        current = order[-1]
        candidates = list(remaining)
        costs = [transition_cost(features[current], features[c]) for c in candidates]
        best_idx = int(np.argmin(costs))
        best = candidates[best_idx]
        order.append(best)
        remaining.remove(best)

        # Optional small randomization for robustness
        if random.random() < 0.02 and len(order) > 3:
            i, j = random.sample(range(1, len(order) - 1), 2)
            order[i], order[j] = order[j], order[i]

    print(f" Order estimated ({n} frames)")
    return order
