import os
import json
import skimage.io
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from selective_search import selective_search
import logging
from tqdm import tqdm
from skimage import img_as_ubyte

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
    def __init__(self,proposal_dir: str):
        self.proposal_dir = proposal_dir
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
    
    # Resources:
        # https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
        # https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        # https://stackoverflow.com/questions/4558835/total-area-of-intersecting-rectangles
    def compute_iou(self, rect1, rect2):
        #intersection over union
               
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Convert (x, y, width, height) to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Compute intersection
        xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])

        x_overlap = xB - xA
        y_overlap = yB - yA

        intersection = x_overlap * y_overlap # area of intersection
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def process_dataset_split(self, split_dir: str, annotation_file: str, output_dir: str):
        """Processes all proposal files in a dataset split."""
        self.logger.info(f"Processing dataset split: {os.path.basename(split_dir)}")

        # Load COCO annotations
        if not os.path.exists(annotation_file):
            self.logger.error(f"Annotation file missing: {annotation_file}")
            return

        with open(annotation_file, "r") as f:
            annotations = json.load(f)

         # Adjusted thresholds
        tp, tn = 0.6, 0.3 

        # Dictionary to store ground truth boxes for quick lookup
        ground_truth_map = {
            image["file_name"]: [
                ann["bbox"] for ann in annotations["annotations"] if ann["image_id"] == image["id"]
            ]
            for image in annotations["images"]
        }

        # Iterate through all proposal files in the split directory
        for proposal_file in os.listdir(split_dir):
            if not proposal_file.endswith(".json"):
                continue  # Skip non-JSON files and the annotations file

            proposal_path = os.path.join(split_dir, proposal_file)

            with open(proposal_path, "r") as f:
                proposals = json.load(f)

            image_filename = proposal_file.replace("_regions.json", ".jpg") # replace the end _regions.json with .jpg to match the image file name in annotations
            ground_truth_boxes = ground_truth_map.get(image_filename, [])

            if not ground_truth_boxes:
                self.logger.warning(f"No ground truth found for {image_filename}, skipping...")
                continue

            positive_samples = []
            negative_samples = []

            # compare proposals with ground truth
            for proposal in proposals:
                max_iou = max([self.compute_iou(proposal["rect"], gt) for gt in ground_truth_boxes], default=0)

                if max_iou >= tp:
                    positive_samples.append(proposal)
                elif max_iou <= tn:
                    negative_samples.append(proposal)

            # Balance negative samples (max 3x positives)
            if positive_samples:
                max_neg = len(positive_samples) * 3
                if len(negative_samples) > max_neg:
                    negative_samples = negative_samples[:max_neg]

            # Save labeled data for this image
            te_output_dir = os.path.join(output_dir, image_filename)
            self.save_regions(positive_samples, os.path.join(te_output_dir, "positive_samples.json"))
            self.save_regions(negative_samples, os.path.join(te_output_dir, "negative_samples.json"))

        

    def run(self, data_root: str, proposal_root: str, output_root: str, splits: list):
        """Runs processing for all dataset splits."""
        for split in splits:
            split_dir_data = os.path.join(data_root, split)
            split_dir_proposal = os.path.join(proposal_root, split)
            annotation_file = os.path.join(split_dir_data, "_annotations.coco.json")
            output_dir = os.path.join(output_root, split)

            self.process_dataset_split(split_dir_proposal, annotation_file, output_dir)


def main():
    base_path = "CV-projectEx5/ex5"
    data_root = os.path.join(base_path, "data/balloon_dataset")
    proposal_root = os.path.join(base_path, "code/results/balloon_regions")
    output_root = os.path.join(base_path, "code/results/balloon_regions/training-examples")
    splits = ['train', 'valid']
    
    
    # Generate training examples
    example_generator = TrainingExampleGenerator(proposal_dir=proposal_root)
    example_generator.run(data_root, proposal_root, output_root, splits)

if __name__ == '__main__':
    main()

