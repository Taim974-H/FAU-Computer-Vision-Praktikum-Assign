import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import convolve
import os
import cv2

def load_data(path, filenames):
    # Create a list to store all images
    raw_images = []
    for name in filenames:
        file_path = os.path.join(path, name)
        raw = rawpy.imread(file_path)
        raw_img = np.array(raw.raw_image_visible)
        raw_images.append(raw_img)  # Append each raw image to the list
    return raw_images  # Return the list of images

def demosaic(raw_img):
    # Padding to handle edge cases
    padded_img = np.pad(raw_img, pad_width=1, mode='symmetric')
    height, width = padded_img.shape

    # Initialize the reconstructed image with the same dimensions as the original
    reconstructed_img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))

    # Create Bayer masks
    red_mask = np.zeros_like(padded_img)
    green_mask = np.zeros_like(padded_img)
    blue_mask = np.zeros_like(padded_img)

    # Create Bayer masks for RGGB pattern
    for i in range(height):   # `i` represents the row index
        for j in range(width):  # `j` represents the column index
            if i % 2 == 0 and j % 2 == 1:
                # Blue channel position in RGGB pattern
                red_mask[i, j] = 1
            elif i % 2 == 1 and j % 2 == 0:
                # Red channel position in RGGB pattern
                blue_mask[i, j] = 1
            else:
                # Green channel position in RGGB pattern
                green_mask[i, j] = 1

    # Separate channels
    red_channel = padded_img * red_mask
    green_channel = padded_img * green_mask
    blue_channel = padded_img * blue_mask

    # Define 3x3 averaging kernel for convolution
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    # Interpolate missing values using convolution
    # For each channel, fill in missing values by averaging surrounding pixels

    # Red channel interpolation
    red_interpolated = convolve(red_channel, kernel, mode='mirror') / convolve(red_mask, kernel, mode='mirror')
    red_interpolated[np.isnan(red_interpolated)] = 0  # Handle divisions by zero

    # Green channel interpolation
    green_interpolated = convolve(green_channel, kernel, mode='mirror') / convolve(green_mask, kernel, mode='mirror')
    green_interpolated[np.isnan(green_interpolated)] = 0

    # Blue channel interpolation
    blue_interpolated = convolve(blue_channel, kernel, mode='mirror') / convolve(blue_mask, kernel, mode='mirror')
    blue_interpolated[np.isnan(blue_interpolated)] = 0

    # Combine channels into the reconstructed image, removing padding
    reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]  # Red
    reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]  # Green
    reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]  # Blue

    return reconstructed_img


def normalize_to_0_1(rgb_img):
    #Normalize the data
    a = np.percentile(rgb_img,0.01)
    b = np.percentile(rgb_img,99.99)
    normalized_img = (rgb_img-a)/(b-a) #shape = (4660, 6984, 3)
    #Clip the values to [0, 1]
    normalized_img[normalized_img < 0] = 0
    normalized_img[normalized_img > 1] = 1
    
    return normalized_img
    
    
def gamma_correction(rgb_img, gamma = 0.3):
    #Gamma Correction
    #applying gamma correction
    img = normalize_to_0_1(rgb_img)
    normalized_gamma_corrected = img**gamma
    
    return normalized_gamma_corrected

def avg_pixel_channel(rgb_image):
    avg_red = np.mean(rgb_image[:,:,0])
    avg_green = np.mean(rgb_image[:,:,1])
    avg_blue = np.mean(rgb_image[:,:,2])
    return avg_red, avg_green, avg_blue
    
def white_balance(rgb_image):
    # Calculate average intensities for each channel
    avg_red, avg_green, avg_blue = avg_pixel_channel(rgb_image)

    # Scaling factors for each channel
    scale_red = avg_green / avg_red
    scale_green = avg_green / avg_green  # This will always be 1
    scale_blue = avg_green / avg_blue

    # Create a copy for processing
    white_balanced_img = rgb_image.copy()

    # Equalize the intensities by scaling each channel
    white_balanced_img[:, :, 0] *= scale_red
    white_balanced_img[:, :, 1] *= scale_green
    white_balanced_img[:, :, 2] *= scale_blue

    # Clip values to avoid overflow
    white_balanced_img = np.clip(white_balanced_img, 0, 1)

    # Scale back to [0, 255] for proper display
    white_balanced_img_display = (white_balanced_img * 255).astype(np.uint8)

    return white_balanced_img_display








def normalize_exposure(raw_images, exposure_times):
    scaled_exposure = []
    longest_exposure = np.max(exposure_times)
    for i in range(len(raw_images)):
        normalized_expo = raw_images[i] * (longest_exposure / exposure_times[i])
        scaled_exposure.append(normalized_expo)
        
    return scaled_exposure

def hdr(scaled_exposure, threshold=0.8):
    """
    Combines multiple exposure images into a single HDR image by replacing
    overexposed pixels with data from shorter exposures.
    """
    # Start with the brightest (longest exposure) image
    hdr_image = scaled_exposure[0].copy()
    max_pixel_value = hdr_image.max()

    # Iterate through shorter exposures
    for next_image in scaled_exposure[1:]:
        # Identify overexposed pixels
        overexposed_mask = hdr_image >= (threshold * max_pixel_value)
        # Replace overexposed pixels with values from the next image
        hdr_image[overexposed_mask] = next_image[overexposed_mask]

    # Handle remaining overexposed pixels using the shortest exposure
    shortest_exposure = scaled_exposure[-1]
    overexposed_mask = hdr_image >= (threshold * max_pixel_value)
    hdr_image[overexposed_mask] = shortest_exposure[overexposed_mask]

    return hdr_image

def tone_map_log(hdr_image):
    # Compute the maximum intensity in the HDR image
    I_max = np.max(hdr_image)
    
    # Apply logarithmic tone mapping
    tone_mapped = np.log(1 + hdr_image) / np.log(1 + I_max)
    
    # Scale to [0, 255] and convert to uint8
    tone_mapped = (tone_mapped * 255).astype(np.uint8)
    return tone_mapped


        
    
    



        
    
    






# Main function
def main():
    # path = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/02"
    # filename = [f"IMG_304{i}.CR3" for i in range(4, 10)]
    
    # # Load raw images
    # raw_images = load_data(path, filename)
    
    # red_means = []
    # green_means = []
    # blue_means = []
    
    # # Process each raw image
    # for idx, raw_img in enumerate(raw_images):
    #     print(f"Processing Image {idx + 1} with shape: {raw_img.shape}")
        
    #     # Perform demosaicing
    #     rgb_image = demosaic(raw_img)
        
    #     # Save the reconstructed image
    #     output_path = os.path.join('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots', f"reconstructed_img_{idx + 1}.png")
    #     plt.imshow(rgb_image.astype(np.uint8))
    #     plt.axis('off')  # Hide axes for cleaner output
    #     plt.title(f"Demosaiced Image {idx + 1}")
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot as an image
    #     plt.close()  # Close the figure to free memory
    #     print(f"Saved Image {idx + 1} to {output_path}")
                

    #     avg_red, avg_green, avg_blue = avg_pixel_channel(rgb_image)
    #     red_means.append(avg_red)
    #     green_means.append(avg_green)
    #     blue_means.append(avg_blue)
        
    
    # exposure_times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320])
    # # Assuming `exposure_times`, `red_means`, `green_means`, and `blue_means` are already defined

    # # Create a figure with 3 subplots (one row, three columns)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # # Red channel
    # axes[0].plot(exposure_times, red_means, color='r', marker="o", label="Red Channel")
    # axes[0].set_title("Red Channel")
    # axes[0].set_xlabel("Exposure Time (seconds)")
    # axes[0].set_ylabel("Average Pixel Value")
    # axes[0].grid()
    # axes[0].legend()

    # # Green channel
    # axes[1].plot(exposure_times, green_means, color='g', marker="o", label="Green Channel")
    # axes[1].set_title("Green Channel")
    # axes[1].set_xlabel("Exposure Time (seconds)")
    # axes[1].set_ylabel("Average Pixel Value")
    # axes[1].grid()
    # axes[1].legend()

    # # Blue channel
    # axes[2].plot(exposure_times, blue_means, color='b', marker="o", label="Blue Channel")
    # axes[2].set_title("Blue Channel")
    # axes[2].set_xlabel("Exposure Time (seconds)")
    # axes[2].set_ylabel("Average Pixel Value")
    # axes[2].grid()
    # axes[2].legend()

    # # Adjust layout for better spacing
    # plt.tight_layout()

    # # Show the plots
    # plt.show()

    # # Save the plot as a single image
    # plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/linearity_sensor_data_subplots.png')
    # plt.close()
    
    #---------------------------------------------------------------------------------------------------------------------------
    #HDR
    path = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/06"
    filename = [f"0{i}.CR3" for i in range(0,10)]
    
    # Load raw images
    raw_images = load_data(path, filename)
    
    rgb_img_list = []
    
    # Initial exposure time
    t0 = 1
    num_images = len(filename)
    
    exposure_times = []

    # Generate the exposure times
    for n in range(num_images):
        t_n = t0 * (1 / 2) ** n  # Calculate the nth exposure time
        exposure_times.append(t_n)  
    
    
    #1 normalize exposure
    normalized_expo_img = normalize_exposure(raw_images, exposure_times)
      
    # # Step 2: Save Normalized Images
    # for idx, img in enumerate(normalize_expo_img):
    #     print(f"Processing Image {idx + 1}")
        
    #     # Normalize pixel values to 0–255 for visualization
    #     img_normalized = img - img.min()  # Ensure minimum is 0
    #     img_normalized = (img_normalized / img_normalized.max()) * 255  # Scale to 0–255
    #     img_normalized = img_normalized.astype(np.uint8)  # Convert to uint8 for saving
        
    #     # Save the image
    #     output_path = os.path.join(
    #         '/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots',
    #         f"normalize_expo_img_{idx + 1}.png"
    #     )
    #     plt.imshow(img_normalized, cmap='gray')  # Use cmap='gray' for grayscale images
    #     plt.axis('off')  # Hide axes for cleaner output
    #     plt.title(f"Normalized Image {idx + 1}")
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot
    #     plt.close()  # Close the figure to free memory
    #     print(f"Saved Image {idx + 1} to {output_path}")
        
    hdr_img = hdr(normalized_expo_img, threshold=0.8) 
    
    # # Step 3: Save HDR Image
    # print("Processing HDR Image")

    # # Normalize pixel values to 0–255 for visualization
    # hdr_normalized = hdr_img - hdr_img.min()  # Ensure minimum is 0
    # hdr_normalized = (hdr_normalized / hdr_normalized.max()) * 255  # Scale to 0–255
    # hdr_normalized = hdr_normalized.astype(np.uint8)  # Convert to uint8 for saving

    # # Define output path
    # output_path = os.path.join(
    #     '/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots',
    #     "hdr_image.png"
    # )

    # # Save and visualize the HDR image
    # plt.imshow(hdr_normalized, cmap='gray')  # Use cmap='gray' for grayscale images
    # plt.axis('off')  # Hide axes for cleaner output
    # plt.title("HDR Image")
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot
    # plt.close()  # Close the figure to free memory

    # print(f"HDR Image saved to {output_path}")
    
    rgb_image = demosaic(hdr_img) 
    gamma_corr_img = gamma_correction(rgb_image,gamma = 0.3)
    white_balanced_img = white_balance(gamma_corr_img)
    tone_mapped_img = tone_map_log(white_balanced_img)

    
        # Display the white-balanced image
    output_path = os.path.join('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots', f"tone_mapped_img.png")
    plt.imshow(tone_mapped_img)
    plt.axis('off')  # Hide axes for cleaner output
    plt.title(f"White Balanced Image")
    plt.savefig(output_path, bbox_inches ='tight', pad_inches=0)  # Save the plot as an image
    plt.close()  # Close the figure to free memory
    print(f"Saved Image to {output_path}")
    
    
    

   
   
        
 #---------------------------------------------------------------------------------------------------------------------   
        # Perform demosaicing
        # rgb_image = demosaic(raw_img)
        
        # # Save the reconstructed image
        # output_path = os.path.join('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots', f"HDR_raw_{idx + 1}.png")
        # plt.imshow(rgb_image.astype(np.uint8))
        # plt.axis('off')  # Hide axes for cleaner output
        # plt.title(f"Demosaiced Image {idx + 1}")
        # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot as an image
        # plt.close()  # Close the figure to free memory
        # print(f"Saved Image {idx + 1} to {output_path}")
        
        # gamma_corr_img = gamma_correction(rgb_image,gamma = 0.3)
        # white_balanced_img = white_balance(gamma_corr_img)
        # rgb_img_list.append(rgb_image)
        
        # # Display the white-balanced image
        # output_path = os.path.join('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots', f"HDR_white_balanced_Image{idx + 1}.png")
        # plt.imshow(white_balanced_img)
        # plt.axis('off')  # Hide axes for cleaner output
        # plt.title(f"White Balanced Image {idx + 1}")
        # plt.savefig(output_path, bbox_inches ='tight', pad_inches=0)  # Save the plot as an image
        # plt.close()  # Close the figure to free memory
        # print(f"Saved Image {idx + 1} to {output_path}")
        
#---------------------------------------------------------------------------------------------------------------------

    
    
    
    

    
        
        
        
        
        
    

        
        
        
        
      

if __name__ == "__main__":
    main()
