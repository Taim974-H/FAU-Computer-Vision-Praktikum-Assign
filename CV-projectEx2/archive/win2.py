import rawpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class RawProcessor:
    def __init__(self):
        pass  

    def process_raw(self, input_path, output_path):
        try:
            # Load image
            raw = rawpy.imread(input_path)
            print(f"Raw image color pattern information:{raw.color_desc}")  

            #demosaicing
            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # Advanced demosaicing method
                use_camera_wb=True,                             # Use the camera's white balance settings
                bright=1.1,                                      # Apply a slight brightness boost
                no_auto_bright=True,                             # Disable automatic brightness adjustment
                output_bps=16,                                   # Output image with 16-bit depth for precision
                gamma=(2.222, 4.5)                              # Apply a gamma correction for a more natural loop                                     # Do not flip the image; set to 0 for no flip
            )
            
            #normalizing
            img = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            print(f"Image shape: {img.shape}")  
            print(f"Value range: [{np.min(img)}, {np.max(img)}]")  
            print(f"Mean value: {np.mean(img)}")
            
         # 1. Contrast Enhancement using CLAHE
            # Convert rgb2LAB
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)  #LAB channel: L = lightness, a = Green–Red and b = Blue–Yellow
            # Apply CLAHE to enhance the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # CLAHE parameters
            l = clahe.apply(l)
            # Convert back LAB2RGB
            lab = cv2.merge([l, a, b])  # Merge the enhanced-L with the unchanged a and b channels
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  


         # 2. Color enhancement(saturation and brightness channels)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to rgb2HSV
            h, s, v = cv2.split(hsv)  # HSV channels: hue, saturation, value/brightness
            # enhance saturation and brightness
            s = cv2.multiply(s, 1.2)
            v = cv2.multiply(v, 1.1)
            # Convert back to RGB
            hsv = cv2.merge([h, s, v]) # Merge the modified channels back
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  

         # 3. edge detection kernel ----> to sharpen details
         # Center --> get a higher value - surrounding pixels --->are reduced
            kernel = np.array([
                [-0.5,-0.5,-0.5], 
                [-0.5, 5.0,-0.5],
                [-0.5,-0.5,-0.5]  
            ])
            #convolving
            img = cv2.filter2D(img, -1, kernel) 

            # 4. Noise reduction
            img = cv2.fastNlMeansDenoisingColored(
                img,                    # Input image
                None,                   # No mask (process all pixels)
                h=5,                    # Denoising strength (adjustable)
                hColor=5,               # Color denoising strength
                templateWindowSize=7,   # Size of the window used for denoising
                searchWindowSize=21     # Size of the search window to find similar patches
            )

            # Save result
            #jpg
            output_path_jpg = os.path.join(output_path, 'IMG_4782.jpg')
            result = cv2.imwrite(output_path_jpg, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save the image (in BGR format)
            #png
            output_path_png = os.path.join(output_path, 'IMG_4782.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)  
            plt.axis('off') 
            plt.title('Processed Image')  
            plt.savefig(output_path_png)
            plt.close()

            print(f"Image saved")
            return img
            
        except:
            print(f"Error processing image:")
            return None
        

# Main
if __name__ == "__main__":
    raw = RawProcessor()
    input_path = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/03/IMG_4782.CR3"   # Data Path
    output_path = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots"       # Folder path to save the images in 
    raw.process_raw(input_path, output_path) 
