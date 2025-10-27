import cv2
import os
from typing import List


def reconstruct_video(frames_dir: str, order: List[str], output_path: str) -> None:
    """
    Combines ordered frames back into a 30 fps MP4 video.
    """
    first_frame = cv2.imread(os.path.join(frames_dir, order[0]))
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    for fname in order:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        out.write(frame)

    out.release()
    print(f"Reconstructed video saved to {output_path}")
