import os
from src.utils import video_downloading, verification_of_specs
from src.frame_extraction import extraction_of_frames
from src.features_extraction import compute_all_features
from src.frame_ordering import final_order_estimation
from src.video_reconstruction import video_reconstruction

Google_drive_link = "https://drive.google.com/file/d/1DbR9yap-vgUaPiI3hCEKUnniXr-TrdOt/view?usp=sharing"
input_path = "data/input_video.mp4"
f_directory = "data/shuffled_frames"
output_path = "data/reconstructed/output.mp4"

def main():
    os.makedirs("data/reconstructed", exist_ok=True)
    os.makedirs("data/shuffled_frames", exist_ok=True)

    if not os.path.exists(input_path):#checking if video is present in local storage
        print("Unable to locate video on local storage, downloading it from google drive...")
        video_downloading(Google_drive_link, input_path)#downloading video if not in local storage from google drive link
    else:
        print("Video already exists on local storage.")

    verification_of_specs(input_path)#confirming the specs of video i.e 30fps and 10 seconds video.

    print("\nFrames extraction initiated...")
    extraction_of_frames(input_path, f_directory)#extracting the frames from the input video.

    print("\nCalculating the frame features...")
    features_all = compute_all_features(f_directory)#frame features are being computed.

    features = {k: v["classic"] for k, v in features_all.items()}

    print("\nOrdering of the frames is under process...")
    order = final_order_estimation(f_directory, features)#frames are being reordered correctly.

    print("\nOriginal video being reconstructed...")
    video_reconstruction(f_directory, order, output_path)#ordered frames are being cnverted to 30 fps and 10 sec video.

    print("Success: original video has been reconstructed.")
    print(f"Recontructed video path:{output_path}")


if __name__ == "__main__":
    main()
