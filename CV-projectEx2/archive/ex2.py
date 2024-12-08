import numpy as np
import rawpy
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path

class HDRProcessor:
    def __init__(self, input_dir):
        """
        Initialize HDR processor with directory containing CR3 files
        Args:
            input_dir (str): Path to directory containing CR3 files
        """
        self.input_dir = input_dir
        self.images = []
        self.exposure_times = []
        self.aligned_images = []
        self.hdr_image = None
        
    def load_images(self):
        """Load CR3 files and their corresponding exposure times"""
        # Get all CR3 files in directory
        cr3_files = sorted(glob.glob(str(Path(self.input_dir) / "*.CR3")))
        
        print(f"Found {len(cr3_files)} CR3 files")
        
        # Calculate exposure times (each image is half the exposure of the previous)
        self.exposure_times = np.array([1 * (0.5**i) for i in range(len(cr3_files))])
        
        # Load each image
        for file in cr3_files:
            with rawpy.imread(file) as raw:
                # Convert to linear RGB, 16-bit, no gamma correction
                rgb = raw.postprocess(gamma=(1,1), 
                                    no_auto_bright=True, 
                                    output_bps=16)
                # Convert to float32 and normalize
                img = rgb.astype(np.float32) / 65535.0
                self.images.append(img)
                
        print(f"Loaded {len(self.images)} images")
        
    def align_images(self):
        """Align images using ECC algorithm"""
        print("Aligning images...")
        
        # Use the middle exposure image as reference
        ref_idx = len(self.images) // 2
        reference = cv2.cvtColor((self.images[ref_idx] * 255).astype(np.uint8), 
                               cv2.COLOR_RGB2GRAY)
        
        self.aligned_images = []
        
        # Parameters for ECC algorithm
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
        
        for idx, img in enumerate(self.images):
            if idx == ref_idx:
                self.aligned_images.append(img)
                continue
                
            # Convert to grayscale for alignment
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            try:
                # Find transformation
                _, warp_matrix = cv2.findTransformECC(reference, gray, 
                                                     warp_matrix, warp_mode, criteria)
                
                # Apply transformation
                aligned = cv2.warpPerspective(img, warp_matrix, 
                                            (img.shape[1], img.shape[0]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                self.aligned_images.append(aligned)
                
            except:
                print(f"Warning: Failed to align image {idx}, using original")
                self.aligned_images.append(img)
                
        print("Alignment complete")
        
    def get_response_curve(self, samples=1000):
        """
        Calculate camera response curve using Debevec's method
        Args:
            samples (int): Number of random pixels to sample
        Returns:
            response_curve (ndarray): Camera response curve
        """
        print("Calculating response curve...")
        
        # Sample random pixels
        h, w = self.aligned_images[0].shape[:2]
        Z = np.zeros((samples, len(self.aligned_images)))
        
        for i, img in enumerate(self.aligned_images):
            # Convert to grayscale
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Sample random pixels
            y_coords = np.random.randint(0, h, samples)
            x_coords = np.random.randint(0, w, samples)
            Z[:, i] = gray[y_coords, x_coords]
            
        # Calculate response curve using Debevec's method
        response_curve = cv2.createCalibrateDebevec()
        response_curve = response_curve.process(Z.astype(np.float32), 
                                              self.exposure_times.astype(np.float32))
        
        return response_curve
        
    def merge_to_hdr(self):
        """Merge aligned images into HDR image"""
        print("Merging to HDR...")
        
        # Create merge_debevec object
        merge_debevec = cv2.createMergeDebevec()
        
        # Convert images to list of uint8 images
        uint8_images = [(img * 255).astype(np.uint8) for img in self.aligned_images]
        
        # Merge images into HDR
        self.hdr_image = merge_debevec.process(uint8_images, 
                                              self.exposure_times.astype(np.float32))
        
        print("HDR merge complete")
        
    def tone_map(self, method='reinhard'):
        """
        Apply tone mapping to HDR image
        Args:
            method (str): Tone mapping method ('reinhard' or 'mantiuk')
        Returns:
            tone_mapped (ndarray): Tone mapped image
        """
        if self.hdr_image is None:
            raise ValueError("No HDR image available. Run merge_to_hdr first.")
            
        print(f"Applying {method} tone mapping...")
        
        if method == 'reinhard':
            tonemap = cv2.createTonemapReinhard(gamma=1.3)
        elif method == 'mantiuk':
            tonemap = cv2.createTonemapMantiuk(gamma=1.3, scale=0.85)
        else:
            raise ValueError(f"Unknown tone mapping method: {method}")
            
        tone_mapped = tonemap.process(self.hdr_image.copy())
        
        # Ensure values are in [0, 1] range
        tone_mapped = np.clip(tone_mapped, 0, 1)
        
        return tone_mapped
        
    def save_result(self, output_path, tone_mapped_image):
        """
        Save tone mapped image
        Args:
            output_path (str): Path to save the result
            tone_mapped_image (ndarray): Tone mapped image to save
        """
        # Convert to uint8
        img_8bit = (tone_mapped_image * 255).astype(np.uint8)
        
        # Save image
        cv2.imwrite(output_path, cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {output_path}")
        
    def process(self, output_path, tone_mapping_method='reinhard'):
        """
        Process HDR pipeline from start to finish
        Args:
            output_path (str): Path to save the final result
            tone_mapping_method (str): Tone mapping method to use
        """
        self.load_images()
        self.align_images()
        self.merge_to_hdr()
        tone_mapped = self.tone_map(method=tone_mapping_method)
        self.save_result(output_path, tone_mapped)
        
        return tone_mapped

# Example usage
if __name__ == "__main__":
    # Initialize processor with directory containing CR3 files
    hdr_processor = HDRProcessor("CV-projectEx2\exercise_2_data\exercise_2_data\\06")
    
    # Process images and save result
    result = hdr_processor.process("output_hdr.jpg")
    
    # Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(result)
    plt.axis('off')
    plt.show()