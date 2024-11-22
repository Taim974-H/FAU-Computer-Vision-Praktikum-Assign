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
          
            raw = rawpy.imread(input_path)
            
            print("Raw image information:")
            print(f"Raw sizes: {raw.sizes}") 
            print(f"Color description: {raw.color_desc}")  
            print(f"Raw pattern: {raw.raw_pattern}") 
             
            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,   
                use_camera_wb=True,                              # Use the camera's white balance settings
                use_auto_wb=False,                               # Do not apply automatic white balance
                no_auto_bright=True,                             # Disable automatic brightness adjustment
                output_bps=16,                                   # Output image with 16-bit depth for precision
                gamma=(2.222, 4.5),                              # Apply a gamma correction for a more natural look
            )
            
            img = rgb.astype(np.float32)
            #normalize the image to 0-255 range
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8) 
            
            print("\nInitial processing stats:")
            print(f"Image shape: {img.shape}")  # image dimensions (height, width, channels)
            print(f"Value range: [{np.min(img)}, {np.max(img)}]")  #range of pixel values
            print(f"Mean value: {np.mean(img)}")
            
            # Perform basic image enhancements:

            # enhancing contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # enhance the L (lightness) channel from LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
            l = clahe.apply(l) 

            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convert back to RGB

            # enhancing saturation and brightness channels
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)  # HSV channels (hue, saturation, value/brightness)
            s = cv2.multiply(s, 1.2)
            v = cv2.multiply(v, 1.1)
            
            hsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # Convert back to RGB

            # sharpening using a custom kernel to improve details (enahnce edges)
            # high value in the center for sharpening and negative values around the center to focus enhancement
            kernel = np.array([
                [-0.5,-0.5,-0.5],
                [-0.5, 5.0,-0.5], 
                [-0.5,-0.5,-0.5]  
            ])
            img = cv2.filter2D(img, -1, kernel) 

            # noise reduction using Non-Local Means Denoising
            img = cv2.fastNlMeansDenoisingColored(
                img,                    # Input image
                None,                   # No mask (process all pixels)
                h=5,                    # Denoising strength (adjustable)
                hColor=5,               # Color denoising strength
                templateWindowSize=7,   # Size of the window used for denoising
                searchWindowSize=21     # Size of the search window to find similar patches
            )

            # histograms of each color channel (Red, Green, Blue) to visualize pixel value distribution
            plt.figure(figsize=(15, 5))
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                plt.subplot(1, 3, i+1)
                plt.hist(img[:,:,i].flatten(), 256, [0,256], color=color, alpha=0.7)  
                plt.title(f'{color.capitalize()} Channel')  
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
            plt.tight_layout()  
            plt.show()  

             
            success = cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
            print(f"\nImage save {'successful' if success else 'failed'}") 
 
            plt.figure(figsize=(12, 8))
            plt.imshow(img) 
            plt.axis('off') 
            plt.title('Processed Image') 
            plt.show()

            return img
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        

# Main execution block
if __name__ == "__main__":
    raw = RawProcessor() 
    input_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\03\IMG_4782.CR3' 
    output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots\IMG_4782.jpg' 
    raw.process_raw(input_path, output_path) 
