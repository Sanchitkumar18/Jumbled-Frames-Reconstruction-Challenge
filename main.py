import os
from src.utils import download_from_gdrive_link, verify_video_specs
from src.frame_extraction import extract_frames
from src.features_extraction import compute_and_cache_features, load_cached_features
from src.frame_ordering import estimate_order
from src.video_reconstruction import reconstruct_video


# ======== Config ========
DRIVE_LINK = "https://drive.google.com/file/d/1DbR9yap-vgUaPiI3hCEKUnniXr-TrdOt/view?usp=sharing"
VIDEO_PATH = "data/input_video.mp4"
FRAMES_DIR = "data/shuffled_frames"
OUTPUT_PATH = "data/reconstructed/output.mp4"
CACHE_PATH = "data/features_cache.pkl"


def main():
    os.makedirs("data/reconstructed", exist_ok=True)
    os.makedirs("data/shuffled_frames", exist_ok=True)

    # Step 1: Download if missing
    if not os.path.exists(VIDEO_PATH):
        print("Video not found locally. Downloading from Google Drive...")
        download_from_gdrive_link(DRIVE_LINK, VIDEO_PATH)
    else:
        print("✅ Video already exists locally.")

    # Step 2: Verify specifications
    verify_video_specs(VIDEO_PATH)

    # Step 3: Extract frames
    print("\n📸 Extracting frames...")
    extract_frames(VIDEO_PATH, FRAMES_DIR)

    # Step 4: Compute or load features
    print("\n🧮 Computing frame features (AI + handcrafted)...")
    features_all = compute_and_cache_features(FRAMES_DIR, cache_path=CACHE_PATH)

    # Use only the handcrafted part for ordering (hybrid model handles fusion internally)
    features = {k: v["classic"] for k, v in features_all.items()}

    # Step 5: Estimate correct order
    print("\n🔍 Estimating correct frame order...")
    order = estimate_order(FRAMES_DIR, features)

    # Step 6: Reconstruct final video
    print("\n🎞️ Reconstructing video...")
    reconstruct_video(FRAMES_DIR, order, OUTPUT_PATH)

    print("\n✅ All steps completed successfully!")
    print(f"🎬 Final output saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
