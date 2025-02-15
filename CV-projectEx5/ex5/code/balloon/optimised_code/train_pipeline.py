"""
Balloon Detection Pipeline using Deep Features and SVM
Author: Taimoor Hussain
Date: 2026-02-14
"""

import os
import json
import sys
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from selective_search import selective_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_IOU_THRESHOLDS = (0.6, 0.4)
BALANCING_RATIO = 2
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

class FeatureExtractor:
    """Deep feature extractor using pretrained CNN models."""
    
    def __init__(self, model_name: str = 'vgg16'):
        """
        Initialize the feature extractor.
        
        Args:
            model_name (str): Name of the pretrained model to use.
        """
        self.model = self._initialize_model(model_name)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
        ])
        
    def _initialize_model(self, model_name: str) -> torch.nn.Module:
        """Initialize pretrained model with last layer removed."""
        model_initializers = {
            'vgg16': models.vgg16,
            'resnet50': models.resnet50,
            'mobilenet_v2': models.mobilenet_v2,
        }
        
        if model_name not in model_initializers:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_initializers.keys())}")
            
        model = model_initializers[model_name](pretrained=True)
        
        if 'resnet' in model_name:
            return torch.nn.Sequential(*list(model.children())[:-1])
        elif 'vgg' in model_name:
            return torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
        return model.features
    
    def extract_features(self, image: Image.Image, region: Dict = None) -> np.ndarray:
        """
        Extract features from an image or a specific region.
        
        Args:
            image (Image.Image): Input image.
            region (Dict): Optional region dictionary with 'rect' key for cropping.
        
        Returns:
            np.ndarray: Extracted feature vector.
        """
        # Crop the image if a region is provided
        if region is not None:
            x, y, w, h = region['rect']
            image = image.crop((x, y, x + w, y + h))
        
        # Transform and extract features
        # Disables gradient computation.
        # During inference (feature extraction), we don’t need to compute gradients because we’re not training the model.
        # Disabling gradients reduces memory usage and speeds up computation.

        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0) #Applies the preprocessing steps (resizing, normalization, etc.) defined in self.transform
            #  .unsqueeze(0) Adds a batch dimension to the tensor. PyTorch models expect inputs in the form (batch_size, channels, height, width).
            #  Without this, the input tensor would have the shape (channels, height, width), which would cause an error.
            
            features = self.model(input_tensor)

            # .squeeze(): Removes the batch dimension (if present) to return a 1D feature vector.
            # .cpu(): Moves the tensor from GPU to CPU (if it was on the GPU).
            # .numpy(): Converts the PyTorch tensor to a NumPy array for compatibility with other libraries (e.g., scikit-learn).
            return features.squeeze().cpu().numpy() # Converts the output tensor into a NumPy array.
        
        
class BalloonDetectionPipeline:
    """Main pipeline for balloon detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._validate_paths()
        self.feature_extractor = FeatureExtractor(model_name='vgg16')
        
    def _validate_paths(self):
        """Validate all required paths exist"""
        required_paths = [
            self.config['testset'],
            self.config['trainset'],
            self.config['valset']
        ]
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Path {path} does not exist")

    def _load_annotations(self, json_path: str) -> Dict:
        """Load COCO format annotations"""
        with open(json_path) as f:
            data = json.load(f)
        
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        return {
            id_to_filename[ann['image_id']]: ann['bbox']
            for ann in data['annotations']
            if ann['image_id'] in id_to_filename
        }

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_min, y1_min, w1, h1 = box1
        x2_min, y2_min, w2, h2 = box2
        
        # Calculate coordinates
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Intersection area
        intersect_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersect_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = intersect_w * intersect_h
        
        # Union area
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def generate_proposals(self):
        """Generate region proposals using selective search"""
        # for dataset in ['train', 'test', 'val']:
        for dataset in ['val']:
            logging.info(f"Processing {dataset} proposals")
            dataset_path = self.config[f'{dataset}set']
            save_dir = Path(self.config['save_path_prp']) / dataset
            save_dir.mkdir(parents=True, exist_ok=True)
            
            bbox_dict = self._load_annotations(self.config[f'{dataset}set_anno_path'])
            
            for img_file in tqdm(bbox_dict.keys(), desc=f"Processing {dataset}"):
                img_path = Path(dataset_path) / img_file
                image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                
                # Selective search implementation should be imported
                _, regions = selective_search(image, scale=300, min_size=15)
                
                output_path = save_dir / f"{img_path.stem}.json"
                with output_path.open('w') as f:
                    json.dump([{'rect': list(map(int, r['rect']))} for r in regions], f)

    def train_classifier(self):
        """Train SVM classifier using selective search proposals with IoU-based labeling."""
        # Load COCO annotations
        with open(self.config['trainset_anno_path'], 'r') as f:
            coco_data = json.load(f)

        # Create image_id to filename mapping
        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        logging.info(f"Processing {len(image_id_to_filename)} images...")

        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann["bbox"])

        # Prepare features and labels
        features, labels = [], []
        proposal_path = Path(self.config['save_path_prp']) / 'train'

        for image_id, gt_boxes in tqdm(annotations_by_image.items(), desc="Processing images"):
            # Get image filename and path
            img_filename = image_id_to_filename[image_id]
            img_path = Path(self.config['trainset']) / img_filename

            # Skip if image doesn't exist
            if not img_path.exists():
                logging.warning(f"Image not found: {img_path}")
                continue

            # Load image
            image = Image.open(img_path).convert('RGB')

            # Load generated proposals
            proposal_file = proposal_path / f"{Path(img_filename).stem}.json"
            if not proposal_file.exists():
                logging.warning(f"Proposal file not found: {proposal_file}")
                continue

            with proposal_file.open() as f:
                regions = json.load(f)

            # Process each proposal
            positive_samples = []
            negative_samples = []

            for region in regions:
                region_box = region['rect']

                # Calculate max IoU with ground truth
                max_iou = max(
                    [self._calculate_iou(region_box, gt) for gt in gt_boxes],
                    default=0.0
                )

                # Label based on IoU thresholds
                if max_iou >= self.config['iou_thresholds'][0]:  # Positive (balloon)
                    positive_samples.append(region)
                elif max_iou <= self.config['iou_thresholds'][1]:  # Negative (background)
                    negative_samples.append(region)

            # Balance classes
            if positive_samples:  # Only balance if we have positive samples
                negative_samples = negative_samples[:len(positive_samples) * BALANCING_RATIO]

                # Extract features for all samples
                for region in positive_samples:
                    features.append(self.feature_extractor.extract_features(image, region))
                    labels.append(1)  # Positive label

                for region in negative_samples:
                    features.append(self.feature_extractor.extract_features(image, region))
                    labels.append(0)  # Negative label

        # Check if we have enough samples
        if not features:
            raise ValueError("No training samples found. Check your data and proposals.")

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train SVM with cross-validation
        svm = SVC(
            kernel='linear',
            probability=True,
            class_weight={0: class_weights[0], 1: class_weights[1]},
            random_state=42
        )

        cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
        logging.info(f"Cross-validation scores: {cv_scores}")

        # Train final model
        svm.fit(X_scaled, y)

        # Save model artifacts
        model_dir = Path(self.config['model_save_path'])
        model_dir.mkdir(exist_ok=True)

        joblib.dump(
            {'svm': svm, 'scaler': scaler},
            model_dir / 'balloon_classifier.pkl'
        )
        logging.info(f"Model saved to {model_dir}")

if __name__ == '__main__':
    config = {
        'testset': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\test",
        'trainset': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\train",
        'valset': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\valid",
        'testset_anno_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\test\_annotations.coco.json",
        'trainset_anno_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\train\_annotations.coco.json",
        'valset_anno_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\valid\_annotations.coco.json",
        'save_path_prp': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\baloon-regions",
        'model_save_path': r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\balloon-regions\models",
        'iou_thresholds': DEFAULT_IOU_THRESHOLDS
    }
    
    pipeline = BalloonDetectionPipeline(config)
    # pipeline.generate_proposals()
    pipeline.train_classifier()