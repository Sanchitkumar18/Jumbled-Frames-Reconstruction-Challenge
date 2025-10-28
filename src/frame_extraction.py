import cv2
import os


def extraction_of_frames(input_path: str, output_path: str) -> None:#extracts all the frames of the input jumbled video.
    os.makedirs(output_path, exist_ok=True)
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        raise FileNotFoundError(f"Error opening given media: {input_path}")

    frames_per_sec = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frames_per_sec == 30, f"30 frames_per_sec were expected, got {frames_per_sec}"
    assert frame_count == 300, f"300 frames were expected, got {frame_count}"

    i = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_path, f"frame_{i:03d}.jpg"), frame)
        i += 1

    capture.release()
    print(f"Extraction of {i} frames has been done to {output_path}")
