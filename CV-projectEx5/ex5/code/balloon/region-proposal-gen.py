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
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class BalloonRegionProposalGenerator:
    def __init__(self, scale: int = 300, min_size: int = 15):  # Adjusted default parameters
        self.scale = scale
        self.min_size = min_size
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
                json.dump(regions, f, cls=NumpyEncoder)
        except Exception as e:
            self.logger.error(f"Error saving regions to {output_path}: {str(e)}")

    def process_dataset_split(self, split_dir: str, annotation_file: str, output_dir: str):
        self.logger.info(f"Processing split: {os.path.basename(split_dir)}")
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        image_map = {
            img['id']: img['file_name'] 
            for img in coco_data['images']
        }
            
        if not image_map:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for image_id, image_name in tqdm(image_map.items(), desc="Generating region proposals"):
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            output_path = os.path.join(output_dir, f"{base_name}_regions.json")
            
            if os.path.exists(output_path):
                continue

            image_path = os.path.join(split_dir, image_name)
            try:
                image = skimage.io.imread(image_path)
            
                if image.dtype != np.uint8:
                    image = img_as_ubyte(image)
                
                _, regions = selective_search(
                    image,
                    scale=self.scale,
                    min_size=self.min_size
                )

                if regions is not None:
                    self.save_regions(regions, output_path)
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                continue

    def run(self, data_root: str, output_root: str, splits: list):
        for split in splits:
            split_dir = os.path.join(data_root, split)
            annotation_file = os.path.join(split_dir, "_annotations.coco.json")
            output_dir = os.path.join(output_root, split)
            self.process_dataset_split(split_dir, annotation_file, output_dir)

def main():
    base_path = "CV-projectEx5/ex5"
    data_root = os.path.join(base_path, "data/balloon_dataset")
    proposal_root = os.path.join(base_path, "code/results/balloon_regions")
    output_root = os.path.join(base_path, "code/results/balloon_regions/training-examples")
    splits = ['train', 'valid']
    
    # Generate region proposals
    generator = BalloonRegionProposalGenerator(
        scale=300,  # Reduced from 500 to get more precise regions
        min_size=15  # Slightly reduced to catch smaller balloons
    )
    generator.run(data_root, proposal_root, splits)
    

if __name__ == '__main__':
    main()