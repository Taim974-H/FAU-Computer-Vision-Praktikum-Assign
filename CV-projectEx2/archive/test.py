import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import convolve
import os
import cv2

def load_data(path, filenames):
    raw_images = []
    for name in filenames:
        try:
            file_path = os.path.join(path, name)
            raw = rawpy.imread(file_path)
            raw_img = np.array(raw.raw_image_visible)
            raw_images.append(raw_img)
        except Exception as e:
            print(f"Error reading {name}: {e}")
            continue
    return raw_images

    
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
            if i % 2 == 0 and j % 2 == 0:
                # Blue channel position in RGGB pattern
                red_mask[i, j] = 1
            elif i % 2 == 1 and j % 2 == 1:
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
    a = np.percentile(rgb_img, 0.01)
    b = np.percentile(rgb_img, 99.99)
    normalized_data = (rgb_img - a) / (b - a)
    normalized_data[normalized_data < 0] = 0
    normalized_data[normalized_data > 1] = 1
    gamma = 0.3
    normalized_gamma_corrected = normalized_data ** gamma
    
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

def hdr_combination(raw_images, exposure_times):
    """
    Combine multiple images to produce HDR raw data
    """
    # Ensure images and exposure times are numpy arrays
    raw_images = np.array(raw_images)
    exposure_times = np.array(exposure_times)
    
    # Weight function: give more weight to middle-range pixel values
    def weight_function(z):
        return 1.0 - np.abs(z - 0.5) * 2.0
    
    # Initialize HDR image as zero
    hdr_image = np.zeros_like(raw_images[0], dtype=np.float64)
    
    # Combine images with their respective weights
    for img, exposure in zip(raw_images, exposure_times):
        # Convert raw image to float
        img_float = img.astype(np.float64)
        # Compute pixel-wise weights
        weights = weight_function(img_float / np.max(img_float))
        # Accumulate weighted image data
        hdr_image += weights * img_float / exposure
    
    return hdr_image

#############################################################

# Main function ------------------------------------------------

#############################################################

def main():
    # Update plots output path to your Windows path
    plots_output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots'

    # Section 2: Demosaicing
    path_2 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\03"
    filename_2 = ["IMG_4782.CR3"]
    raw_img_2 = load_data(path_2, filename_2)
    
    # Demosaic IMG_4782
    reconstructed_img = demosaic(raw_img_2[0])
    
    # Visualize reconstructed image
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_img.astype(np.uint8))
    plt.title("Reconstructed RGB Image via Interpolation")
    plt.colorbar()
    plt.savefig(os.path.join(plots_output_path, 'reconstructed_img.png'))
    plt.close()

    # Section 3: Improve Luminosity
    normalized_img = normalize_0_to_1(reconstructed_img)
    luminosity_corr_img = gamma_correction(normalized_img, gamma=0.3)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(luminosity_corr_img)
    plt.title("Gamma Corrected Image")
    plt.axis("off")
    plt.savefig(os.path.join(plots_output_path, 'Gamma_Corrected_Image.png'))
    plt.close()

    # Section 4: White Balance
    balanced_img = white_balance(luminosity_corr_img)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(balanced_img)
    plt.title("White Balanced Image")
    plt.axis("off")
    plt.savefig(os.path.join(plots_output_path, 'white_balanced_Image_display.png'))
    plt.close()

    #############################################################

    # Section 5: Sensor Data Linearity

    #############################################################

    
    path_5 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\02"
    filename_5 = [f"IMG_304{i}.CR3" for i in range(4, 10)]
    raw_img_5 = load_data(path_5, filename_5)
    
    exposure_times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320])
    red_means, green_means, blue_means = [], [], []
    
    for idx, raw_img in enumerate(raw_img_5):
        print(f"Processing Image {idx + 1} ...")
        rgb_image = demosaic(raw_img)
        avg_red, avg_green, avg_blue = avg_pixel_channel(rgb_image)
        red_means.append(avg_red)
        green_means.append(avg_green)
        blue_means.append(avg_blue)
    
    # Plot linearity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Red channel
    axes[0].plot(exposure_times, red_means, color='r', marker="o", label="Red Channel")
    axes[0].set_title("Red Channel")
    axes[0].set_xlabel("Exposure Time (seconds)")
    axes[0].set_ylabel("Average Pixel Value")
    axes[0].grid()
    axes[0].legend()

    # Green channel
    axes[1].plot(exposure_times, green_means, color='g', marker="o", label="Green Channel")
    axes[1].set_title("Green Channel")
    axes[1].set_xlabel("Exposure Time (seconds)")
    axes[1].set_ylabel("Average Pixel Value")
    axes[1].grid()
    axes[1].legend()

    # Blue channel
    axes[2].plot(exposure_times, blue_means, color='b', marker="o", label="Blue Channel")
    axes[2].set_title("Blue Channel")
    axes[2].set_xlabel("Exposure Time (seconds)")
    axes[2].set_ylabel("Average Pixel Value")
    axes[2].grid()
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_path, 'linearity_sensor_data_subplots.png'))
    plt.close()

    #############################################################

    # Section 6: HDR Implementation

    #############################################################

    # path_6 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\06"
    # filename_6 = [f"0{i}.CR3" for i in range(0,10)]
    
    # raw_img_6 = load_data(path_6, filename_6)
    
    # # Exposure times
    # t0 = 1
    # num_images = len(filename_6)
    # exposure_times_6 = [t0 * (1 / 2) ** n for n in range(num_images)]
    
    # # Normalize exposure
    # normalized_expo_img = normalize_exposure(raw_img_6, exposure_times_6)
    
    # hdr_img = hdr(normalized_expo_img, threshold=0.8) 
    
    # # Log tone mapping
    # log_hdr = np.log(1 + hdr_img)
    # max_log_hdr = np.max(hdr_img)
    # normalized_hdr = (log_hdr / max_log_hdr * 255).astype(np.uint8)

    # hdr_rgb_image = demosaic(normalized_hdr) 
    # gamma_corr_img = gamma_correction(hdr_rgb_image, gamma=0.3)
    # white_balanced_img = white_balance(gamma_corr_img)

    

    # # Visualize HDR image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(white_balanced_img)
    # plt.title("HDR White Balanced Image")
    # plt.axis("off")
    # plt.savefig(os.path.join(plots_output_path, 'hdr_white_balanced_img.png'))
    # plt.close()
    
    path_6 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\06"
    filename_6 = [f"0{i}.CR3" if i < 10 else f"{i}.CR3" for i in range(0, 11)]
    
    
    # Load raw images
    raw_img_6 = load_data(path_6, filename_6)
    # Save the result
    plots_output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots'

    
    # Exposure times: halving for each subsequent image
    t0 = 1
    exposure_times = [t0 * (1/2)**n for n in range(len(filename_6))]
    
        # Combine images and create HDR raw data
    hdr_img = hdr_combination(raw_img_6, exposure_times)

    # Demosaic HDR raw image first
    hdr_rgb_image = demosaic(hdr_img)
 
    # Plot HDR Processing Steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # HDR Raw Image (first image before processing)
    axes[0, 0].imshow(hdr_img)
    axes[0, 0].set_title('HDR Raw Image')
    axes[0, 0].axis('off')

    # Demosaiced RGB Image
    axes[0, 1].imshow(hdr_rgb_image.astype(np.uint8))
    axes[0, 1].set_title('Demosaiced HDR RGB')
    axes[0, 1].axis('off')

    # Log HDR Image
    log_hdr = np.log(1 + hdr_rgb_image)
    axes[0, 2].imshow((log_hdr / np.max(log_hdr) * 255).astype(np.uint8))
    axes[0, 2].set_title('Log Tone Mapped')
    axes[0, 2].axis('off')

    # Normalized HDR
    normalized_hdr = (log_hdr / np.max(log_hdr) * 255).astype(np.uint8)
    axes[1, 0].imshow(normalized_hdr)
    axes[1, 0].set_title('Normalized HDR')
    axes[1, 0].axis('off')

    # Gamma Correction
    gamma_corr_img = gamma_correction(normalized_hdr/255.0, gamma=0.3)
    axes[1, 1].imshow((gamma_corr_img * 255).astype(np.uint8))
    axes[1, 1].set_title('Gamma Corrected')
    axes[1, 1].axis('off')

    # White Balance
    white_balanced_img = white_balance(gamma_corr_img)
    axes[1, 2].imshow(white_balanced_img)
    axes[1, 2].set_title('White Balanced')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_path, 'hdr_processing_steps.png'))
    plt.close()

if __name__ == "__main__":
    main()