import os
import time
from src.utils import video_downloading, verification_of_specs
from src.frame_extraction import extraction_of_frames
from src.features_extraction import compute_all_features
from src.frame_ordering import final_order_estimation
from src.video_reconstruction import video_reconstruction


Google_drive_link = "https://drive.google.com/file/d/1DbR9yap-vgUaPiI3hCEKUnniXr-TrdOt/view?usp=sharing"
input_path = "data/input_video.mp4"
f_directory = "data/shuffled_frames"
output_path = "data/reconstructed/output.mp4"
log_path = "data/log_report.txt"


def log_time(step_name: str, start_time: float, log_file):
    """Log report for every step."""
    end_time = time.time()
    duration = end_time - start_time
    msg = f"{step_name}: {duration:.2f} seconds"
    print(f"{msg}")
    log_file.write(msg + "\n")
    log_file.flush()
    return end_time


def main():
    os.makedirs("data/reconstructed", exist_ok=True)
    os.makedirs("data/shuffled_frames", exist_ok=True)

    total_start = time.time()

    # Keep the log file open until all steps are done
    with open(log_path, "w") as log_file:
        log_file.write("Execution Time Log\n\n")

        # Step 1: Download or verify existing video
        start_time = time.time()
        if not os.path.exists(input_path):
            print("Unable to locate video on local storage, downloading from Google Drive...")
            video_downloading(Google_drive_link, input_path)
        else:
            print("Video already exists on local storage.")
        log_time("Download & Verification", start_time, log_file)

        # Step 2: Verify video specs
        verification_of_specs(input_path)

        # Step 3: Extract frames
        print("\nFrames extraction initiated...")
        start_time = time.time()
        extraction_of_frames(input_path, f_directory)
        log_time("Frame Extraction", start_time, log_file)

        # Step 4: Compute features
        print("\nCalculating the frame features...")
        start_time = time.time()
        features_all = compute_all_features(f_directory)
        log_time("Feature Computation", start_time, log_file)

        features = {k: v["classic"] for k, v in features_all.items()}

        # Step 5: Estimate order
        print("\nOrdering of the frames is under process...")
        start_time = time.time()
        order = final_order_estimation(f_directory, features)
        log_time("Frame Ordering", start_time, log_file)

        # Step 6: Reconstruct video
        print("\nOriginal video being reconstructed...")
        start_time = time.time()
        video_reconstruction(f_directory, order, output_path)
        log_time("Video Reconstruction", start_time, log_file)

        # Step 7: Total time
        total_time = time.time() - total_start
        msg = f"\nTotal Execution Time: {total_time:.2f} seconds"
        print("\nSuccess: Original video has been reconstructed.")
        print(f"Reconstructed video path: {output_path}")

        log_file.write(msg + "\n")
        log_file.flush()


if __name__ == "__main__":
    main()
