import numpy as np
from typing import Dict, List
from tqdm import tqdm

def transition_cost(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Transition cost = Euclidean distance + brightness difference penalty.
    """
    diff = np.linalg.norm(v1 - v2)
    brightness_diff = abs(np.mean(v1[:16]) - np.mean(v2[:16]))
    return diff + 0.5 * brightness_diff

def greedy_sequence(features: Dict[str, np.ndarray], start_frame: str) -> List[str]:
    """
    Builds a sequence greedily from a start frame.
    """
    frame_names = list(features.keys())
    remaining = set(frame_names)
    order = [start_frame]
    remaining.remove(start_frame)

    while remaining:
        current = order[-1]
        best = min(remaining, key=lambda f: transition_cost(features[current], features[f]))
        order.append(best)
        remaining.remove(best)
    return order

def smooth_sequence(order: List[str], features: Dict[str, np.ndarray], window: int = 5) -> List[str]:
    """
    Local optimization: swap adjacent frames if it reduces total cost.
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            current_cost = transition_cost(features[order[i]], features[order[i+1]])
            swapped_cost = transition_cost(features[order[i+1]], features[order[i]])
            if swapped_cost + 0.001 < current_cost:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
    return order

def estimate_order(frames_dir: str, features: Dict[str, np.ndarray]) -> List[str]:
    """
    Estimates frame order using bidirectional greedy search and smoothing.
    """
    frame_names = list(features.keys())

    # Start points: darkest and brightest frames
    darkest = min(frame_names, key=lambda f: np.mean(features[f][:16]))
    brightest = max(frame_names, key=lambda f: np.mean(features[f][:16]))

    print("Estimating frame order (bidirectional)...")
    forward_order = greedy_sequence(features, darkest)
    backward_order = list(reversed(greedy_sequence(features, brightest)))

    # Compare both and pick the smoother one
    def total_cost(order):
        return sum(transition_cost(features[order[i]], features[order[i+1]])
                   for i in range(len(order)-1))

    if total_cost(backward_order) < total_cost(forward_order):
        order = backward_order
    else:
        order = forward_order

    # Apply local smoothing
    order = smooth_sequence(order, features)

    print(f"Final order estimated ({len(order)} frames).")
    return order
