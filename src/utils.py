import requests
import re
import cv2


def googledrive_file_id(Link: str) -> str:
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]#extracting the file id from google drive link
    for p in patterns:
        match = re.search(p, Link)
        if match:
            return match.group(1)
    raise ValueError("failed to fetch the file id from the given google drive link")


def video_downloading(Link: str, destination: str) -> None:
   
    file_id = googledrive_file_id(Link)
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

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

    print(f"Success: video is downloaded to {destination}")


def verification_of_specs(Path: str) -> None:
    capture = cv2.VideoCapture(Path)
    frames_per_sec = int(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / frames_per_sec if frames_per_sec else 0
    capture.release()

    print(f" video specifications → Frames_per_sec: {frames_per_sec}, Total_frames: {frames}, Duration: {duration:.2f}s")

    if frames_per_sec != 30 or frames != 300:
        raise ValueError(" Video is not of 10s or 30 fps ,it should have 300 frames")
    else:
        print(" Success:Video is verified")
