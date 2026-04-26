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
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
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
os.makedirs(FEATURES_DIR, exist_ok=True)
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

def sliding_window(image, step_size, window_size):
    """Slides a window across the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def non_max_suppression(boxes, overlapThresh):
    """Removes overlapping bounding boxes, keeping the highest confidence ones."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        # Delete indices that overlap too much
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype(int)

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
    np.save(os.path.join(FEATURES_DIR, 'X_features.npy'), X)
    np.save(os.path.join(FEATURES_DIR, 'y_labels.npy'), y)
    print("Saved dataset to 'X_features.npy' and 'y_labels.npy'.")



    print("Loading extracted HOG dataset...")
    try:
        X = np.load(os.path.join(FEATURES_DIR, 'X_features.npy'))
        y = np.load(os.path.join(FEATURES_DIR, 'y_labels.npy'))
    except FileNotFoundError:
        print("Error: Could not find X_features.npy or y_labels.npy.")
        print("Please run the extraction script first.")
        return

    print(f"Dataset loaded! Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")

    # 1. Split the data into Training (80%) and Testing (20%) sets
    # 'stratify=y' ensures the 80/20 split maintains the same ratio of stop signs to backgrounds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 2. Initialize the Linear Support Vector Classifier
    # max_iter is set high to ensure the math converges properly
    model = LinearSVC(max_iter=10000, random_state=42)

    # 3. Train the model
    print("\nTraining the SVM classifier... (This might take a few seconds)")
    model.fit(X_train, y_train)

    # 4. Evaluate the model on the unseen Test data
    print("Evaluating model performance on test set...")
    y_pred = model.predict(X_test)

    # 5. Print the results
    print("-" * 40)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Background (0)', 'Stop Sign (1)']))

    # 6. Save the trained model to a file
    joblib.dump(model, MODEL_FILE)
    print("-" * 40)
    print(f"Success! Model saved to '{MODEL_FILE}'.")



    print(f"Loading model '{MODEL_FILE}'...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Error: Model file not found. Run train_model.py first.")
        return

    print(f"Loading test image '{TEST_IMAGE}'...")
    image = io.imread(TEST_IMAGE)
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image

    detections = []
    current_scale = 1.0
    scaled_image = gray_image

    print("Scanning image... (This might take a minute depending on image size)")

    # --- THE IMAGE PYRAMID ---
    while scaled_image.shape[0] >= WINDOW_SIZE[1] and scaled_image.shape[1] >= WINDOW_SIZE[0]:

        # --- THE SLIDING WINDOW ---
        for (x, y, window) in sliding_window(scaled_image, STEP_SIZE, WINDOW_SIZE):

            # Extract HOG for this specific window
            # MUST match the exact parameters from your training script!
            hog_feat = feature.hog(window,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   block_norm='L2-Hys',
                                   transform_sqrt=True,
                                   feature_vector=True)

            # SVM uses decision_function for confidence scores
            score = model.decision_function(hog_feat.reshape(1, -1))[0]

            if score > CONFIDENCE_THRESHOLD:
                # Convert the window coordinates back to the original image scale
                orig_x = int(x * current_scale)
                orig_y = int(y * current_scale)
                orig_w = int(WINDOW_SIZE[0] * current_scale)
                orig_h = int(WINDOW_SIZE[1] * current_scale)

                detections.append([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h, score])

        # Shrink the image for the next pyramid level
        scaled_image = transform.rescale(scaled_image, 1.0 / SCALE_FACTOR, anti_aliasing=True)
        current_scale *= SCALE_FACTOR

    # Clean up overlapping boxes using NMS
    final_boxes = non_max_suppression(detections, overlapThresh=0.3)

    print(f"Found {len(final_boxes)} stop sign(s) after Non-Maximum Suppression!")

    # --- DRAW THE RESULTS ---
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for (startX, startY, endX, endY, score) in final_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((startX, startY), endX - startX, endY - startY,
                                 linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(startX, startY - 10, f"Stop Sign: {score/100:.2f}", color='red', fontsize=12, weight='bold')

    plt.title("Stop Sign Detection (HOG + SVM)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()