import os
import json
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extract_train_svm import FeatureExtractorResNet50 as FeatureExtractor

class BalloonDetector:
    def __init__(self, model_path="balloon_detector_svm.pkl"):
        """Initialize the balloon detector with a trained SVM model."""
        # Load the trained SVM model and scaler
        model_data = joblib.load(model_path)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.feature_extractor = FeatureExtractor()

    def detect_balloons(self, image_path, proposals_dir):
        """Detect balloons in a single image using the trained SVM"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get image filename for proposals path
        image_filename = os.path.basename(image_path).replace(".jpg", "_regions.json")
        proposals_path = os.path.join(proposals_dir, image_filename)
        
        if not os.path.exists(proposals_path):
            print(f"No proposals found for {proposals_path}")
            return image, []
        
        # Load region proposals
        with open(proposals_path, 'r') as f:
            proposals = json.load(f)
        detections = []
        
        # Process each region proposal
        for region in proposals:
            x, y, w, h = region['rect']
            
            # Crop and extract features
            cropped_region = image.crop((x, y, x + w, y + h))
            features = self.feature_extractor.extract_features(cropped_region)
            
            # Reshape and scale features
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and confidence
            prediction = self.svm.predict(features_scaled)[0]
            confidence = self.svm.predict_proba(features_scaled)[0][1]
            
            # Store if predicted as balloon with high confidence
            if prediction == 1 and confidence >= 0.80:  # subject to change - adjusted to 0.7 for now
                detections.append({
                    'rect': [x, y, w, h],
                    'confidence': float(confidence)
                })
        
        return image, detections

    def visualize_detections(self, image, detections, save_path=None):
        """Visualize detected balloons on the image"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Draw each detection
        for det in detections:
            x, y, w, h = det['rect']
            conf = det['confidence']
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Add confidence score
            plt.text(
                x, y-5,
                f'{conf:.2f}',
                color='red',
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def main():

    # Workflow: 
    # 1. Load the trained SVM model and scaler
    # 2. Load the test set annotations to get the image list
    # 3. Process each image in the test set
    # 4. Detect balloons in each image using the trained SVM
    # 5. Save the detection results to the output directory
    
    base_path = os.path.abspath("C:/Users/taimo/Desktop/computer-vision-project/CV-projectEx5/ex5")    
    model_path = os.path.join(base_path, r"code\balloon\balloon_detector_svm.pkl")
    test_image_dir = os.path.join(base_path, "data/balloon_dataset/test")
    proposals_dir = os.path.join(base_path, "code/results/balloon_regions/test")
    output_dir = os.path.join(base_path, "code/results/balloon_regions/detection_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = BalloonDetector(model_path)
    
    # Load test set annotations to get image list
    annotations_path = os.path.join(test_image_dir, "_annotations.coco.json")
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)
    
    # Process each image
    for img_data in coco_data['images']:
        image_filename = img_data['file_name']
        image_path = os.path.join(test_image_dir, image_filename)
        
        print(f"Processing {image_filename}...")
        
        # Detect balloons (using region proposals and trained SVM)
        image, detections = detector.detect_balloons(image_path, proposals_dir)
        
        # Save results
        output_path = os.path.join(output_dir, f"detection_{image_filename}")
        detector.visualize_detections(image, detections, save_path=output_path)
        
        print(f"Found {len(detections)} balloons")
        print(f"Saved detection result to {output_path}\n")

if __name__ == "__main__":
    main()