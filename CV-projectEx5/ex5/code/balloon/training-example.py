import numpy as np
import json
import os
import logging
import skimage.io
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class TrainingExampleGenerator:
    def __init__(self, image_dir: str, annotation_file: str):

        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.setup_logging()

    def setup_logging(self): 
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def save_regions(self, regions: dict, output_path: str):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(regions, f, cls=NumpyEncoder) # used custom encoder for NumPy types to fix serialization error
        except Exception as e:
            self.logger.error(f"Error saving regions to {output_path}: {str(e)}")

    def compute_iou(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Convert (x, y, width, height) to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Compute intersection
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

    def process_dataset_split(self, split_dir: str, annotation_file: str, output_dir: str):
        self.logger.info(f"Processing split: {os.path.basename(split_dir)}")
        
        with open("proposals.json", "r") as f:
                proposals = json.load(f)

        # load COCO annotations
        with open(annotation_file, 'r') as f:
                annotations = json.load(f)

        ground_truth_boxes = [obj["bbox"] for obj in annotations["annotations"]]  # Extract ground truth boxes
        tp, tn = 0.75, 0.25  # Thresholds
        positive_samples = []
        negative_samples = []
        


def main():
    # Configuration
    base_path = "CV-projectEx5/ex5"
    data_root = os.path.join(base_path, "/code/results/balloon_regions")
    output_root = os.path.join(base_path, "code/results/balloon_regions/training-examples")
    splits = ['train', 'valid']
    
    # Initialize generator
    generator = TrainingExampleGenerator()
    
    # Process each split
    for split in splits:
        split_dir = os.path.join(data_root, split)
        annotation_file = os.path.join(split_dir, "_annotations.coco.json")
        output_dir = os.path.join(output_root, split)
        
        generator.process_dataset_split(split_dir, annotation_file, output_dir)

if __name__ == '__main__':
    main()