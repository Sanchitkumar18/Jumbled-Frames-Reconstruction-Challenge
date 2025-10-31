

# Jumbled Frames Reconstruction Challenge

## Overview

This project reconstructs a temporally jumbled video back to its original correct order by analyzing visual continuity between frames.
It uses image processing and feature similarity-based optimization to estimate the most probable frame sequence, then reassembles them into a smooth 10-second video at 30 frames per second.

The main objective is to restore temporal order from a randomly shuffled set of frames by leveraging color, brightness, texture, and edge continuity.

---

## Features

* Automatically downloads and verifies the input video from a Google Drive link
* Extracts individual frames from the video
* Computes feature vectors for every frame using brightness, color, and edge information
* Estimates the correct temporal order using a cost-minimization algorithm
* Reconstructs the final ordered video at 30 FPS and 10 seconds duration
* Generates a detailed log report with the execution time for each step

---

## Project Structure

```
Jumbled-Frames-Reconstruction-Challenge/
│
├── main.py                        # Main pipeline (handles all stages sequentially)
│
├── src/
│   ├── frame_extraction.py        # Extracts frames from the input video
│   ├── features_extraction.py     # Computes a compact feature descriptor for each frame
│   ├── frame_ordering.py          # Orders frames based on visual similarity
│   ├── video_reconstruction.py    # Combines ordered frames into a video
│   └── utils.py                   # Helper utilities (downloading, verification, etc.)
│
├── data/
│   ├── shuffled_frames/           # Extracted or shuffled frames
│   ├── reconstructed/             # Final reconstructed output video
│   └── log_report.txt             # Step-wise and total execution time report
│
├── requirements.txt
└── README.md
```

---

## Algorithm Details

The algorithm consists of five main stages.

### Step 1: Frame Extraction

The input video is read using OpenCV and split into individual frames.
Each frame is stored as an image inside the `data/shuffled_frames` directory.
A typical 10-second video at 30 FPS produces exactly 300 frames.

---

### Step 2: Feature Extraction

Each frame is converted into a fixed-size feature vector that captures its essential visual characteristics.
These features help in comparing frames and determining their correct sequential order.

The feature vector is 26-dimensional and includes the following components:

1. **Brightness Grid**:
   The frame is converted to grayscale and divided into a 4x4 grid (16 regions).
   The mean brightness of each region is computed, producing 16 brightness values.

2. **Edge Intensity**:
   Using the Sobel operator, the average edge magnitude across the frame is calculated.
   This captures motion or sharp transitions between frames.

3. **Mean Hue**:
   The image is converted to HSV color space, and the average hue value is computed.
   This helps track gradual color shifts between frames.

4. **Gradient Orientation Histogram**:
   A histogram with 8 bins is generated to represent the distribution of edge directions.
   This helps detect motion patterns and object orientation consistency.

The resulting 16 + 1 + 1 + 8 = 26 features per frame effectively summarize lighting, texture, and motion cues.

---

### Step 3: Frame Similarity Calculation

For each pair of frames (A, B), a similarity cost is computed based on the difference between their feature vectors.
The lower the cost, the more likely the two frames are adjacent in time.

The cost function used is:

```
Cost(A, B) = ||features(A) - features(B)||_2 + 0.5 * |Brightness(A) - Brightness(B)|
```

Here:

* `||features(A) - features(B)||_2` represents the Euclidean distance between the feature vectors.
* The brightness difference term helps ensure smooth lighting transitions between consecutive frames.

This cost function penalizes abrupt visual changes, which are usually caused by incorrect frame ordering.

---

### Step 4: Frame Ordering

The ordering process is a combination of heuristic and optimization-based methods.

1. **Initialization**
   The algorithm identifies two extreme frames:

   * The darkest frame (usually the first frame)
   * The brightest frame (often the last frame)

2. **Greedy Construction**
   Starting from the darkest frame, the algorithm repeatedly chooses the next frame that has the minimum transition cost compared to the current frame.

3. **Bidirectional Check**
   The process is repeated in reverse (starting from the brightest frame backward), creating two candidate sequences.

4. **Sequence Comparison**
   Both forward and backward sequences are evaluated for total transition cost.
   The sequence with the lower cumulative cost is selected.

5. **Local Optimization**
   A final refinement is applied where adjacent frames are swapped if the swap decreases the total transition cost.
   This ensures local smoothness and corrects small ordering mistakes.

The result is a near-optimal temporal ordering of frames.

---

### Step 5: Video Reconstruction

The ordered frames are stitched together into a continuous video using OpenCV’s `cv2.VideoWriter`.
The output video is saved at 30 frames per second and a total duration of 10 seconds.
The final video is stored as `data/reconstructed/output.mp4`.

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sanchitkumar18/Jumbled-Frames-Reconstruction-Challenge.git
cd Jumbled-Frames-Reconstruction-Challenge
```

### Step 2: Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate      # For macOS / Linux
venv\Scripts\activate         # For Windows
pip install -r requirements.txt
```

### Step 3: Run the Main Script

```bash
python main.py
```

When executed, the program performs the following:

1. Downloads the jumbled video from the Google Drive link defined in `main.py`.
2. Verifies that the video duration is 10 seconds and frame rate is 30 FPS.
3. Extracts all video frames.
4. Computes the features for each frame.
5. Estimates the correct frame order.
6. Reconstructs the ordered video.

The final reconstructed video is saved in the following location:

```
data/reconstructed/output.mp4
```

The detailed log file (execution time per stage) is stored at:

```
data/log_report.txt
```

---

## Testing with a New Jumbled Video

You can easily test this code with any new video.

1. Upload your new jumbled video to Google Drive.
2. Make sure the link is shareable to “Anyone with the link can view.”
3. Copy the Google Drive link (for example:
   `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
4. Open `main.py` and replace the value of `Google_drive_link` with your new link:

```python
Google_drive_link = "https://drive.google.com/file/d/YOUR_NEW_FILE_ID/view?usp=sharing"
```

5. Run the script again:

```bash
python main.py
```

The program will automatically download the new video, process it, and reconstruct it.

---

## Example Log Output

```
Execution time log...

Download & Verification: 2.34 seconds
Frame Extraction: 3.81 seconds
Feature Computation: 4.92 seconds
Frame Ordering: 5.47 seconds
Video Reconstruction: 1.96 seconds

Total time Taken for Execution: 18.50 seconds
```

---

## Technical Details

* Language: Python 3.9 or higher
* Libraries Used: OpenCV, NumPy, Torch, tqdm, Pillow, scikit-image, scipy
* Input Format: MP4, 10 seconds, 30 FPS
* Output Format: MP4 (Reconstructed, 10 seconds, 30 FPS)
* Platform Support: macOS, Windows, Linux

---


## Author

Name: Sanchit Kumar

Institution: VIT Vellore

Project: TECDIA Internship Challenge

Email: [[hcpssanchit@gmail.com](mailto:your-email@example.com)]

---


