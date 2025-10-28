import cv2
import numpy as np
import os
import pickle as p
import torch as t
import torchvision.models as models
import torchvision.transforms as Transforms
from PIL import Image as I
from typing import Dict as D
from tqdm import tqdm

device = t.device("cuda" if t.cuda.is_available() else "cpu")#used to check if device has cuda-compatible gpu or not.
resnet = models.resnet18(pretrained=True)#this is used to create a CNN architecture and to load pretrained datasets.
resnet = t.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

transform = Transforms.Compose([
    Transforms.Resize((224, 224)),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


#mathematical aaproach to calculate brightness grid,Edge intensity,hue,and histogram
def compute_features(Path: str) -> np.ndarray:
    
    img = cv2.imread(Path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converting bg to grey
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#converting to hue staturation value

    grey = cv2.equalizeHist(grey)#normalise brightness

    h, w = grey.shape
    grid = [np.mean(grey[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]) / 255.0
            for i in range(4) for j in range(4)]
    grid = np.array(grid)#brightness grid

    vertical = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)#sobel operator used to detect edges of the image(vertical)
    horizontal = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)#(horizontal)
    magnitude = np.sqrt(vertical**2 + horizontal**2)
    strength_of_edge = np.mean(magnitude)/255.0

    ang_in_deg = np.degrees(np.arctan2(horizontal, vertical))%180
    histogram, _ = np.histogram(ang_in_deg, bins=8, range=(0, 180), weights=magnitude)
    histogram = histogram / (np.sum(histogram) + 1e-6)#histogram gradient calculation


    hue = np.mean(hsv[:, :, 0])/180.0#mean hue calculation

    extracted_features = np.concatenate([grid, [strength_of_edge, hue], histogram])
    return extracted_features.astype(np.float32)


def compute_other_features(Path: str) -> np.ndarray:
    try:
        img = I.open(Path).convert("RGB")
        with t.no_grad():
            tensor = transform(img).unsqueeze(0).to(device)
            feature = resnet(tensor).squeeze().cpu().numpy()
        feature /= np.linalg.norm(feature) + 1e-8
        return feature.astype(np.float32)
    except Exception as e:
        print(f"feature extraction failed :{Path}: {e}")
        return np.zeros(512, dtype=np.float32)



def compute_all_features(frames: str) -> D[str, D[str, np.ndarray]]:
    features = {}
    f_file = sorted([frame for frame in os.listdir(frames)
                          if frame.lower().endswith((".jpg", ".png"))])

    print(f"features extraction of {len(f_file)} frames is under progress...")
    for fname in tqdm(f_file, desc="extracting"):
        path = os.path.join(frames, fname)
        features[fname] = {
            "classic": compute_features(path),
            "other": compute_other_features(path)
        }

    print("Success: Feature extarction is done")
    return features