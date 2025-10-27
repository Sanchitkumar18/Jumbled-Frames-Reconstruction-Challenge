import cv2
import os


def extract_frames(video_path: str, out_dir: str) -> None:
    """
    Extracts all frames from a 10-second (30 fps) video into individual images.

    Args:
        video_path: Path to the input video (must be 10 s @ 30 fps).
        out_dir: Directory to save the extracted frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps == 30, f"Expected 30 fps, got {fps}"
    assert frame_count == 300, f"Expected 300 frames, got {frame_count}"

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:03d}.jpg"), frame)
        i += 1

    cap.release()
    print(f"Extracted {i} frames to {out_dir}")
