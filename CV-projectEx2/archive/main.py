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

    
    # #only for Task 2 to see how is the post processed RGB image
    # rgb_image = raw.postprocess()
    # plt.imshow(rgb_image)
    # plt.title("Demosaiced RGB Image")
    # plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/rgb_image.png')
    # plt.close()

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
    kernel = np.ones((3,3))

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


def normalize_0_to_1(rgb_img):
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
    img = normalize_0_to_1(rgb_img)
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


def visualize(plots_output_path,filename,rgb_image,idx): 
    # Save the reconstructed image
    output_path = os.path.join(plots_output_path, f"{filename}_{idx + 1}.png")
    plt.imshow(rgb_image.astype(np.uint8))
    plt.axis('off')  # Hide axes for cleaner output
    plt.title(f"{filename} {idx + 1}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot as an image
    plt.close()  # Close the figure to free memory
    print(f"Saved Image {idx + 1} to {output_path}")

#--------------------------------------------------------------------------------
# visualization functions


    
    

# # Main function
def main():
    plots_output_path = '/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots'
# #-----------------------------------------------------------------------------------------------------------------------------------
# #2 Implement a Demosaicing Algorithm on IMG_4782

#     path_2 = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/03"
#     filename_2 = [f"IMG_4782.CR3"]

#     # Load raw image data (Bayer pattern)
#     raw_img_2 = load_data(path_2,filename_2)

# # # Display the Bayer pattern as a grayscale image
# #     plt.imshow(raw_img_2[0], cmap='gray')
# #     plt.title("Bayer Pattern Data")
# #     plt.colorbar()
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/raw_img.png')
# #     plt.close()

#     #demosaic IMG_4782
#     reconstructed_img = demosaic(raw_img_2[0])


# # # Display the reconstructed image
# #     plt.imshow(reconstructed_img.astype(np.uint8))
# #     plt.title("Reconstructed RGB Image via Interpolation")
# #     plt.colorbar()
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/reconstructed_img.png')
# #     plt.close()

# # # optional: plotted section of IMG
# #     section_reconst_img = reconstructed_img[:10,:10]
# #     plt.imshow(section_reconst_img.astype(np.uint8))
# #     plt.title("section_reconst_img RGB Image via Interpolation")
# #     plt.colorbar()
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/section_reconst_img.png')
# #     plt.close()


# # If you wanna see the post processed rgb image go the the load_data() func. and uncomment the visualization



# #-------------------------------------------------------------------------------------------------------------------------------------
#  #3 Improve the Luminosity 
#     normalized_img = normalize_0_to_1(reconstructed_img)
#     Luminosity_corr_img = gamma_correction(normalized_img, gamma = 0.3)

# # #Plot the gamma-corrected image
# #     plt.figure(figsize=(8, 8))
# #     plt.imshow(Luminosity_corr_img)
# #     plt.title("Gamma Corrected Image")
# #     plt.axis("off")
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/Gamma_Corrected_Image.png')
# #     plt.close()
# # #-------------------------------------------------------------------------------------------------------------------------------------
# #4 White Balance
#     balanced_img = white_balance(Luminosity_corr_img)

# # #Display the white-balanced image
# #     plt.figure(figsize=(8, 8))
# #     plt.imshow(balanced_img)
# #     plt.title("White Balanced Image")
# #     plt.axis("off")
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/white_balanced_Image_display.png')
# #     plt.close()

# # #-----------------------------------------------------------------------------------------------------------------------------------
# #5 Show that Sensor Data is Linear
#     path_5 = "/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/02"
#     filename_5 = [f"IMG_304{i}.CR3" for i in range(4, 10)]

#     # Load raw images
#     raw_img_5 = load_data(path_5, filename_5)
    
#     exposure_times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320])
#     red_means = []
#     green_means = []
#     blue_means = []
    
#     # Process each raw image
#     for idx, raw_img in enumerate(raw_img_5):
#         print(f"Processing Image {idx + 1} ...")
        
#         # Perform demosaicing
#         rgb_image = demosaic(raw_img)

#         # visualize reconstructed image
#         # visualize(plots_output_path,filename="reconstructed_img",rgb_image,idx)

#         avg_red, avg_green, avg_blue = avg_pixel_channel(rgb_image)
#         red_means.append(avg_red)
#         green_means.append(avg_green)
#         blue_means.append(avg_blue)
        
    
# # # subplots to show linearity of sensor data
# #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# #     # Red channel
# #     axes[0].plot(exposure_times, red_means, color='r', marker="o", label="Red Channel")
# #     axes[0].set_title("Red Channel")
# #     axes[0].set_xlabel("Exposure Time (seconds)")
# #     axes[0].set_ylabel("Average Pixel Value")
# #     axes[0].grid()
# #     axes[0].legend()

# #     # Green channel
# #     axes[1].plot(exposure_times, green_means, color='g', marker="o", label="Green Channel")
# #     axes[1].set_title("Green Channel")
# #     axes[1].set_xlabel("Exposure Time (seconds)")
# #     axes[1].set_ylabel("Average Pixel Value")
# #     axes[1].grid()
# #     axes[1].legend()

# #     # Blue channel
# #     axes[2].plot(exposure_times, blue_means, color='b', marker="o", label="Blue Channel")
# #     axes[2].set_title("Blue Channel")
# #     axes[2].set_xlabel("Exposure Time (seconds)")
# #     axes[2].set_ylabel("Average Pixel Value")
# #     axes[2].grid()
# #     axes[2].legend()

# #     # Adjust layout for better spacing
# #     plt.tight_layout()

# #     # Show the plots
# #     plt.show()

# #     # Save the plot as a single image
# #     plt.savefig('/home/cip/medtech2022/ed16eteh/Documents/Computer_Vision/exercise_2/exercise_2_data/plots/linearity_sensor_data_subplots.png')
# #     plt.close()
    
# #---------------------------------------------------------------------------------------------------------------------------
#6 Initial HDR Implementation
    path_6 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\06"
    filename_6 = [f"0{i}.CR3" for i in range(0,10)]
    
    # Load raw images
    raw_img_6 = load_data(path_6, filename_6)
    
    rgb_img_list = []
    
    #exposure time
    t0 = 1
    num_images = len(filename_6)
    
    exposure_times_6 = []

    # Generate the exposure times
    for n in range(num_images):
        t_n = t0 * (1 / 2) ** n  # Calculate the nth exposure time
        exposure_times_6.append(t_n)  
    
    
    # normalize exposure
    normalized_expo_img = normalize_exposure(raw_img_6, exposure_times_6)
      
    # # Save Normalized Images
    # for idx, img in enumerate(normalized_expo_img):
    #     print(f"Processing Image {idx + 1}...")
        
    #     # Normalize pixel values to 0–255 for visualization
    #     img_normalized = img - img.min()  # Ensure minimum is 0
    #     img_normalized = (img_normalized / img_normalized.max()) * 255  # Scale to 0–255
    #     img_normalized = img_normalized.astype(np.uint8)  # Convert to uint8 for saving
        
        # Save the image
        # visualize(plots_output_path=plots_output_path,filename="normalized_expo_img",rgb_image = img_normalized,idx =idx)

        
    hdr_img = hdr(normalized_expo_img, threshold=0.8) 
    
    # Normalize pixel values to 0–255 for visualization
    hdr_normalized = hdr_img - hdr_img.min()  # Ensure minimum is 0
    hdr_normalized = (hdr_normalized / hdr_normalized.max()) * 255  # Scale to 0–255
    hdr_normalized = hdr_normalized.astype(np.uint8)  # Convert to uint8 for saving

    # # Save the image
    # visualize(plots_output_path,filename="hdr_image",rgb_image =hdr_normalized,idx=0)
    
    hdr_rgb_image = demosaic(hdr_img) 
    gamma_corr_img = gamma_correction(hdr_rgb_image,gamma = 0.3)
    white_balanced_img = white_balance(gamma_corr_img)

    # visualize(plots_output_path,filename="white_balanced_img",rgb_image =white_balanced_img,idx=0)
           
# #---------------------------------------------------------------------------------------------------------------------
    

if __name__ == "__main__":
    main()
