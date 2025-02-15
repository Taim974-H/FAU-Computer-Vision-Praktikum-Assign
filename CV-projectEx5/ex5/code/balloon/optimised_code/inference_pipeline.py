import os
import json
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from train_pipeline import FeatureExtractor

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DETECTION_COLOR = 'red'
GROUND_TRUTH_COLOR = 'green'


def load_svm_model(model_path: str) -> tuple:
    """
    Load the trained SVM model and scaler from a file.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        tuple: (svm, scaler) model and scaler objects.
    """
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
    return model_data['svm'], model_data['scaler']


def load_annotations(json_path: str) -> dict:
    """
    Load COCO format annotations and create a mapping of image IDs to filenames and bounding boxes.
    
    Args:
        json_path (str): Path to the COCO annotations file.
    
    Returns:
        dict: Dictionary mapping image IDs to (filename, bboxes) tuples.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create image ID to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Create dictionary mapping image IDs to (filename, bboxes)
    gt_dict = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in gt_dict:
            gt_dict[image_id] = {
                'filename': id_to_filename[image_id],
                'bboxes': []
            }
        gt_dict[image_id]['bboxes'].append(ann['bbox'])
    
    return gt_dict


def detect_balloons(
    image_path: str,
    proposals_path: str,
    svm,
    scaler,
    feature_extractor,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> tuple:
    """
    Detect balloons in an image using region proposals and a trained SVM classifier.
    
    Args:
        image_path (str): Path to the input image.
        proposals_path (str): Path to the region proposals JSON file.
        svm: Trained SVM classifier.
        scaler: StandardScaler used during training.
        feature_extractor: Feature extractor for region proposals.
        confidence_threshold (float): Minimum confidence score for detection.
    
    Returns:
        tuple: (image, detected_rects) where detected_rects is a list of bounding boxes.
    """
    image = Image.open(image_path).convert("RGB")
    with open(proposals_path, 'r') as f:
        proposals = json.load(f)

    detected_rects = []
    for region in proposals:
        # Extract features for the region
        features = feature_extractor.extract_features(image, region).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict and get confidence score
        y_pred = svm.predict(features_scaled)[0]
        conf_score = svm.predict_proba(features_scaled)[0][1]
        
        # Filter based on prediction and confidence
        if y_pred == 1 and conf_score >= confidence_threshold:
            detected_rects.append({'rect': region['rect'], 'confidence': conf_score})

    return image, detected_rects


def draw_bounding_box(
    image: Image.Image,
    detections: list,
    ground_truths: list,
    save_path: str,
    image_filename: str
) -> None:
    """
    Draw bounding boxes on the image and save the result.
    
    Args:
        image (Image.Image): Input image.
        detections (list): List of detected bounding boxes.
        ground_truths (list): List of ground truth bounding boxes.
        save_path (str): Directory to save the output image.
        image_filename (str): Name of the input image file.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Draw detections
    for det in detections:
        x, y, w, h = det['rect']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=4, edgecolor=DETECTION_COLOR, facecolor='none', label="Detection"
        )
        plt.gca().add_patch(rect)

    # Draw ground truths
    for gt in ground_truths:
        x, y, w, h = gt
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=4, edgecolor=GROUND_TRUTH_COLOR, facecolor='none', label="Ground Truth"
        )
        plt.gca().add_patch(rect)

    plt.axis('off')
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"detection_{Path(image_filename).stem}.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()


def main(config: dict):
    # Load model and feature extractor
    model_path = os.path.join(config['model_save_path'], 'balloon_classifier.pkl')
    svm, scaler = load_svm_model(model_path)
    feature_extractor = FeatureExtractor(model_name='vgg16')

    # Load ground truth annotations
    gt_dict = load_annotations(config['testset_anno_path'])

    # Process each image in the test set
    for image_id, data in tqdm(gt_dict.items(), desc="Processing images"):
        filename = data['filename']
        bboxes = data['bboxes']

        # Construct paths using actual filename
        image_path = os.path.join(config['testset'], filename)
        proposals_path = os.path.join(config['save_path_prp'], 'test', f"{Path(filename).stem}.json")

        if not os.path.exists(proposals_path):
            print(f"Proposals not found for {filename}, skipping...")
            continue

        # Detect balloons
        image, detections = detect_balloons(image_path, proposals_path, svm, scaler, feature_extractor)

        # Visualize and save results
        draw_bounding_box(image, detections, bboxes, config['save_path_prp'], filename)
        print(f"Found {len(detections)} balloons in {filename}")


if __name__ == "__main__":
    # Configuration
    
    config = {
        'testset': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\test",
        'testset_anno_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\test\_annotations.coco.json",
        'save_path_prp': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\zainab-regions",
        'model_save_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\zainab-regions\models",
    }

    # Run inference pipeline
    main(config)