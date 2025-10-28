import cv2
import os
from typing import List as L


def video_reconstruction(f_directory: str, order: L[str], output_path: str) -> None:#formation of new 30fps video after combining ordered frames.
    
    first_frame = cv2.imread(os.path.join(f_directory, order[0]))
    height, width, _ = first_frame.shape
    four_character_code = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, four_character_code, 30, (width, height))

    for fname in order:
        frame = cv2.imread(os.path.join(f_directory, fname))
        out.write(frame)

    out.release()
    print(f"Reconstructed video saved to {output_path}")
