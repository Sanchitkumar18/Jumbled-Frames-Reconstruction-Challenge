import os
from src.utils import download_from_gdrive_link, verify_video_specs
from src.frame_extraction import extract_frames
from src.features_extraction import load_features
from src.frame_ordering import estimate_order
from src.video_reconstruction import reconstruct_video



DRIVE_LINK = "https://drive.google.com/file/d/1DbR9yap-vgUaPiI3hCEKUnniXr-TrdOt/view?usp=sharing"
VIDEO_PATH = "data/input_video.mp4"
FRAMES_DIR = "data/shuffled_frames"
OUTPUT_PATH = "data/reconstructed/output.mp4"



def main():
    os.makedirs("data/reconstructed", exist_ok=True)
    os.makedirs("data/shuffled_frames", exist_ok=True)

    # Download video if missing
    if not os.path.exists(VIDEO_PATH):
        print("Video not found locally. Downloading from Google Drive...")
        download_from_gdrive_link(DRIVE_LINK, VIDEO_PATH)
    else:
        print(" Video already exists locally.")

    # Verify specifications (10s @ 30fps)
    verify_video_specs(VIDEO_PATH)

    # Extract frames
    print("\n Extracting frames...")
    extract_frames(VIDEO_PATH, FRAMES_DIR)

    #Compute features
    print("\n Computing frame features...")
    features = load_features(FRAMES_DIR)

    # Estimate correct order
    print("\n Estimating correct frame order...")
    order = estimate_order(FRAMES_DIR, features)

    #  Reconstruct final video
    print("\n Reconstructing video...")
    reconstruct_video(FRAMES_DIR, order, OUTPUT_PATH)

    print("\n All steps completed successfully!")
    print(f"Final output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
