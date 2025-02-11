import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

class FeatureExtractorResNet50:
    def __init__(self, model_name="resnet50"):

        if model_name == "resnet50": # (working with resnet50 for now)
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            # removed final classification/fully connected layer to output only feature vectors/embeddings
            self.feature_layer = model.fc.in_features
            model.fc = nn.Identity() 

        # disables dropout and batch normalization updates. This means that the model will run in inference mode
        # inference mode means that the model will not update the weights

        self.model = model.eval()  # Set to evaluation mode. eval() disables dropout and batch normalization updates -> consisetnt f.e. extraction
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match CNN input size as that is the resent input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization values for resnet50. These valuse are of the mean and standard deviation of the ImageNet dataset
        ])

    def extract_features(self, image):
        """Extract features from an image (PIL format)."""

        # .unsqueeze(0) adds a batch dimension (ResNet expects input as [batch_size, channels, height, width])
        transformed_image = self.transform(image).unsqueeze(0)
        with torch.no_grad(): # Disable gradient tracking. This will reduce memory consumption for computations that would otherwise have `requires_grad=True`.
            features = self.model(transformed_image)
        return features.squeeze().numpy() # .squeeze() removes unnecessary dimensions (e.g., [1, 2048] → [2048])

def load_training_samples(proposal_root, image_dir):
    """Loads positive and negative samples for SVM training."""
    X, Y = [], [] # Feature vectors and labels

    extractor = FeatureExtractorResNet50()


    annotations_path = os.path.join(image_dir, "_annotations.coco.json")
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Image ID -> Filename mapping
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    print(f"Processing {len(image_id_to_filename)} images...")

    for image_id, file_name in image_id_to_filename.items():
        image_path = os.path.join(image_dir, file_name)

        # loading image
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping...")
            continue  
        image = Image.open(image_path).convert("RGB")

        # Locate positive/negative sample files
        sample_dir = os.path.join(proposal_root, file_name)
        pos_json_path = os.path.join(sample_dir, "positive_samples.json")
        neg_json_path = os.path.join(sample_dir, "negative_samples.json")

        # Load regions from JSON files
        def load_regions(json_path):
            return json.load(open(json_path)) if os.path.exists(json_path) else []

        positive_regions = load_regions(pos_json_path)
        negative_regions = load_regions(neg_json_path)

        # Extract features from positive samples
        for region in positive_regions:
            x, y, w, h = region["rect"]
            cropped_region = image.crop((x, y, x + w, y + h)) # we need cropped region from image to extract features (of the balloons)
            features = extractor.extract_features(cropped_region)
            X.append(features)
            Y.append(1)  # Positive label

        # Extract features from negative samples
        for region in negative_regions:
            x, y, w, h = region["rect"]
            cropped_region = image.crop((x, y, x + w, y + h))
            features = extractor.extract_features(cropped_region)
            X.append(features)
            Y.append(0)  # Negative label

    print(f"Loaded {len(X)} samples")
    return np.array(X), np.array(Y)

def train_svm(X, y, save_path="balloon_detector_svm.pkl"):

    # Ensure X is a 2D array before scaling
    X = X.reshape(X.shape[0], -1)
    # svm sensitive to feature scales (larger values might be given more importance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel="linear", probability=True)  # linear svm
    svm.fit(X_scaled, y)

    joblib.dump({"svm": svm, "scaler": scaler}, save_path)
    print(f"SVM model saved to {save_path}")
    return svm

def main():
    base_path = os.path.abspath("C:/Users/taimo/Desktop/computer-vision-project/CV-projectEx5/ex5")
    proposal_root = os.path.join(base_path, "code/results/balloon_regions/training-examples/train")  # Paths for proposal regions and image dataset
    image_dir = os.path.join(base_path, "data/balloon_dataset/train")

    if not os.path.exists(proposal_root):
        print(f"Proposal root directory not found: {proposal_root}")
        return

    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return

    # Extract training samples
    X, Y = load_training_samples(proposal_root, image_dir)

    # Train SVM if samples are available
    if len(X) > 0:
        train_svm(X, Y)
    else:
        print("No training samples found. Cannot train SVM.")

if __name__ == "__main__":
    main()