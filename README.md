# Stop Sign Detector (HOG + Linear SVM)

This project implements a custom object detection pipeline designed to identify stop signs in images. Unlike modern deep learning approaches (such as YOLO), this project uses traditional computer vision techniques, making it **lightweight**, **interpretable**, and easy to experiment with.

## Features

- **Automatic dataset generation:** Parses XML annotations to extract positive (stop sign) and negative (background) image patches.
- **Feature extraction:** Uses Histogram of Oriented Gradients (HOG) to capture structural and edge information.
- **Machine learning model:** Trains a Linear SVM classifier using scikit-learn.
- **Multi-scale detection:** Uses an image pyramid and sliding window to detect stop signs at different sizes.
- **Post-processing:** Applies Non-Maximum Suppression (NMS) to remove redundant overlapping detections.

## Prerequisites

Ensure Python is installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## File Structure

The script expects a structure similar to:

```text
project-root/
|
+-- data/
|   +-- images/            # Source .jpg/.png images
|   +-- annotations.xml    # CVAT/Pascal VOC style XML file
|   +-- test_stop.png      # Example test image
+-- main.py                # Main script

## How It Works

### 1. Data Preparation

The script reads `annotations.xml` to locate stop sign bounding boxes.

- **Positives:** Crops stop sign regions and resizes them to `64x64`.
- **Negatives:** Randomly crops background regions that do not overlap with stop signs.

### 2. Training

The model extracts HOG features from all crops and trains a `LinearSVC`. The trained model is saved as `stop_sign_svm.pkl`.

### 3. Inference (Detection)

When running on a test image, the script:

1. Rescales the image at multiple levels (image pyramid).
2. Slides a `64x64` window across each scale.
3. Classifies each window.
4. Suppresses overlapping boxes to keep only the most confident detections.

## Configuration

You can tweak these constants at the top of the script to improve performance:

| Variable | Description | Default |
| --- | --- | --- |
| `STEP_SIZE` | Pixels the window moves per step. Lower values are more accurate but slower. | `16` |
| `SCALE_FACTOR` | How much the image shrinks in the pyramid. | `1.25` |
| `CROP_SIZE` | The internal resolution the SVM sees. | `(64, 64)` |
| `CONFIDENCE_THRESHOLD` | Sensitivity of the detector. | `0.0` |

## Results and Visualization

After execution, the script displays the test image with red bounding boxes around detected stop signs and prints a classification report with **Precision**, **Recall**, and **Accuracy**.