import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from skimage import io, transform, color, feature
from skimage.util import img_as_ubyte
import warnings
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Suppress warnings about saving low-contrast images
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
XML_FILE = os.path.join(DATA_DIR, 'annotations.xml')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
POS_DIR = os.path.join(DATA_DIR, 'positives')
NEG_DIR = os.path.join(DATA_DIR, 'negatives')
CROP_SIZE = (64, 64)
NEGATIVES_PER_IMAGE = 15 # Will generate 15 * 50 = 750 negative samples
MODEL_FILE = 'stop_sign_svm.pkl'
TEST_IMAGE = os.path.join(DATA_DIR, 'test_stop.jpg') # Replace with an image from your folder
WINDOW_SIZE = (64, 64)
STEP_SIZE = 16  # How many pixels the window moves at a time
SCALE_FACTOR = 1.25 # How much to shrink the image at each pyramid level
CONFIDENCE_THRESHOLD = 0.0 # Minimum confidence score to be considered a detection

# Create output directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

def check_overlap(box1, box2):
    """Checks if two bounding boxes overlap. Box format: (xmin, ymin, xmax, ymax)"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return False # No overlap
    return True # Overlap exists

def extract_hog_from_folder(image_dir, label):
    """Reads images from a folder, extracts HOG features, and assigns a label."""
    features_list = []
    labels_list = []

    # Check if folder exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        return [], []

    # Loop through all images in the folder
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, filename)
        image = io.imread(img_path)

        # 1. Convert to grayscale
        # (HOG focuses on structure and edges, so color is usually discarded)
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # 2. Extract HOG features
        hog_feat = feature.hog(image,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L2-Hys', # Standard normalization
                               transform_sqrt=True, # Reduces the effect of heavy shadows
                               feature_vector=True) # Flattens the output into a 1D array

        features_list.append(hog_feat)
        labels_list.append(label)

    return features_list, labels_list


def main():
    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    pos_count = 0
    neg_count = 0

    print("Starting extraction...")

    for image_elem in root.findall('image'):
        img_name = image_elem.get('name')
        img_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Could not find image {img_path}. Skipping.")
            continue

        # Load image
        image = io.imread(img_path)
        img_h, img_w = image.shape[:2]

        ground_truth_boxes = []

        # --- 1. EXTRACT POSITIVE SAMPLES ---
        for box_elem in image_elem.findall('box'):
            if box_elem.get('label') == 'stop_sign':
                # Parse coordinates (converting from string to float to int)
                xtl = int(float(box_elem.get('xtl')))
                ytl = int(float(box_elem.get('ytl')))
                xbr = int(float(box_elem.get('xbr')))
                ybr = int(float(box_elem.get('ybr')))

                ground_truth_boxes.append((xtl, ytl, xbr, ybr))

                # Crop the positive sample
                pos_crop = image[ytl:ybr, xtl:xbr]

                # Resize to 64x64
                pos_resized = transform.resize(pos_crop, CROP_SIZE, anti_aliasing=True)

                # Save it
                save_path = os.path.join(POS_DIR, f"pos_{pos_count:04d}.jpg")
                io.imsave(save_path, img_as_ubyte(pos_resized))
                pos_count += 1

        # --- 2. EXTRACT NEGATIVE SAMPLES ---
        attempts = 0
        negatives_extracted = 0

        # We loop until we get enough negatives, or we fail too many times
        while negatives_extracted < NEGATIVES_PER_IMAGE and attempts < 100:
            attempts += 1

            # Generate a random box size (simulate different background scales)
            size = random.randint(64, max(65, min(img_h, img_w) // 2))

            # Generate random top-left coordinates
            x1 = random.randint(0, img_w - size)
            y1 = random.randint(0, img_h - size)
            x2 = x1 + size
            y2 = y1 + size

            # Ensure this random box does NOT overlap with any stop sign
            overlap = False
            for gt_box in ground_truth_boxes:
                if check_overlap((x1, y1, x2, y2), gt_box):
                    overlap = True
                    break

            if not overlap:
                # Crop and resize the negative sample
                neg_crop = image[y1:y2, x1:x2]
                neg_resized = transform.resize(neg_crop, CROP_SIZE, anti_aliasing=True)

                # Save it
                save_path = os.path.join(NEG_DIR, f"neg_{neg_count:04d}.jpg")
                io.imsave(save_path, img_as_ubyte(neg_resized))
                neg_count += 1
                negatives_extracted += 1

    print("-" * 30)
    print(f"Done! Extracted {pos_count} positive samples and {neg_count} negative samples.")






    print("Extracting HOG features from positive samples (Stop Signs)...")
    pos_features, pos_labels = extract_hog_from_folder(POS_DIR, label=1)

    print("Extracting HOG features from negative samples (Backgrounds)...")
    neg_features, neg_labels = extract_hog_from_folder(NEG_DIR, label=0)

    if len(pos_features) == 0 or len(neg_features) == 0:
        print("Extraction failed: Missing images in one or both directories.")
        return

    # Combine positive and negative data
    X_list = pos_features + neg_features
    y_list = pos_labels + neg_labels

    # Convert lists to Numpy arrays (the format scikit-learn requires)
    X = np.array(X_list)
    y = np.array(y_list)

    print("-" * 30)
    print("Feature extraction complete!")
    print(f"Shape of X (Features matrix): {X.shape}")
    print(f"Shape of y (Labels vector): {y.shape}")

    # Save the dataset to disk so we don't have to recalculate HOG later
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
    print("Saved dataset to 'X_features.npy' and 'y_labels.npy'.")


if __name__ == '__main__':
    main()