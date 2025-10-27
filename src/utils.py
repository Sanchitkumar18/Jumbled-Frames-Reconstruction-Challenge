import requests
import re
import cv2


def extract_file_id(drive_link: str) -> str:
    """
    Extracts the file ID from a Google Drive link.
    Works for both formats:
        - https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
        - https://drive.google.com/open?id=<FILE_ID>
    """
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        match = re.search(p, drive_link)
        if match:
            return match.group(1)
    raise ValueError("Could not extract file ID from the provided Google Drive link.")


def download_from_gdrive_link(drive_link: str, destination: str) -> None:
    """
    Downloads a file directly from a Google Drive sharing link.
    Example:
        download_from_gdrive_link(
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing",
            "data/input_video.mp4"
        )
    """
    file_id = extract_file_id(drive_link)
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Handle download confirmation for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f" Download complete → {destination}")


def verify_video_specs(video_path: str) -> None:
    """
    Ensures the input video is exactly 10 seconds @ 30 fps (≈300 frames).
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps else 0
    cap.release()

    print(f" Video info → FPS: {fps}, Frames: {frames}, Duration: {duration:.2f}s")

    if fps != 30 or frames != 300:
        raise ValueError(" Video must be 10 seconds @ 30 fps (300 frames).")
    else:
        print(" Video specification verified: 10s @ 30fps.")
