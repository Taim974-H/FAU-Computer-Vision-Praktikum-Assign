import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import convolve
import os
import cv2
from PIL import Image, ImageDraw, ImageFont

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

# Demosaicing ------------------------------------------------

def demosaic(raw_img):
    # Padding to handle edge cases
    padded_img = np.pad(raw_img, pad_width=1, mode='symmetric')
    height, width = padded_img.shape

    # Initialize the reconstructed image
    reconstructed_img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))

    # Create Bayer masks (adjust the pattern if needed)
    red_mask = np.zeros_like(padded_img)
    green_mask = np.zeros_like(padded_img)
    blue_mask = np.zeros_like(padded_img)

    # for i in range(height):
    #     for j in range(width):
    #         if i % 2 == 0 and j % 2 == 0:  # RGGB pattern
    #             red_mask[i, j] = 1
    #         elif i % 2 == 1 and j % 2 == 1:
    #             blue_mask[i, j] = 1
    #         else:
    #             green_mask[i, j] = 1

    # Create Bayer masks for BGGR pattern
    for i in range(height):
        for j in range(width):
            if i % 2 == 0 and j % 2 == 0:  # Top-left (Blue)
                blue_mask[i, j] = 1
            elif i % 2 == 1 and j % 2 == 1:  # Bottom-right (Red)
                red_mask[i, j] = 1
            else:  # Green
                green_mask[i, j] = 1

    # Separate channels
    red_channel = padded_img * red_mask
    green_channel = padded_img * green_mask
    blue_channel = padded_img * blue_mask

    # Avoid division by zero during interpolation
    kernel = np.ones((3, 3))
    red_interpolated = np.divide(
        convolve(red_channel, kernel, mode='mirror'),
        convolve(red_mask, kernel, mode='mirror'),
        out=np.zeros_like(red_channel, dtype=np.float64),
        where=convolve(red_mask, kernel, mode='mirror') != 0
    )
    green_interpolated = np.divide(
        convolve(green_channel, kernel, mode='mirror'),
        convolve(green_mask, kernel, mode='mirror'),
        out=np.zeros_like(green_channel, dtype=np.float64),
        where=convolve(green_mask, kernel, mode='mirror') != 0
    )
    blue_interpolated = np.divide(
        convolve(blue_channel, kernel, mode='mirror'),
        convolve(blue_mask, kernel, mode='mirror'),
        out=np.zeros_like(blue_channel, dtype=np.float64),
        where=convolve(blue_mask, kernel, mode='mirror') != 0
    )

    # Combine channels into the reconstructed image
    reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]
    reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]
    reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]

    # Normalize and convert to uint8
    reconstructed_img -= reconstructed_img.min()
    reconstructed_img /= reconstructed_img.max()
    reconstructed_img = np.clip(reconstructed_img * 255, 0, 255).astype(np.uint8)

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

# def tone_map_log(hdr_image):
#     # Compute the maximum intensity in the HDR image
#     I_max = np.max(hdr_image)
    
#     # Apply logarithmic tone mapping
#     tone_mapped = np.log(1 + hdr_image) / np.log(1 + I_max)
    
#     # Scale to [0, 255] and convert to uint8
#     tone_mapped = (tone_mapped * 255).astype(np.uint8)
#     return tone_mapped


def visualize(plots_output_path,filename,rgb_image,idx): 
    # Save the reconstructed image
    output_path = os.path.join(plots_output_path, f"{filename}_{idx + 1}.png")
    plt.imshow(rgb_image.astype(np.uint8))
    plt.axis('off')  # Hide axes for cleaner output
    plt.title(f"{filename} {idx + 1}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the plot as an image
    plt.close()  # Close the figure to free memory
    print(f"Saved Image {idx + 1} to {output_path}")




def create_hdr_plot_pillow(
    hdr_img, hdr_rgb_image, normalized_hdr, gamma_corr_img_uint8, white_balanced_img, output_file
):
    # Ensure images are in the right format for Pillow
    hdr_img_pillow = Image.fromarray((hdr_img / hdr_img.max() * 255).astype(np.uint8))  # Convert HDR to grayscale
    hdr_rgb_pillow = Image.fromarray(hdr_rgb_image.astype(np.uint8))  # Convert RGB to Pillow
    normalized_hdr_pillow = Image.fromarray(normalized_hdr.astype(np.uint8))  # Grayscale
    gamma_corr_pillow = Image.fromarray(gamma_corr_img_uint8.astype(np.uint8))  # RGB
    white_balanced_pillow = Image.fromarray(white_balanced_img.astype(np.uint8))  # RGB

    # Image size and padding
    img_width, img_height = hdr_img_pillow.size
    padding = 20
    grid_width = 3  # 3 columns
    grid_height = 2  # 2 rows

    # Create a blank canvas for the plot
    canvas_width = grid_width * img_width + (grid_width + 1) * padding
    canvas_height = grid_height * img_height + (grid_height + 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))  # White background

    # Add Titles
    titles = [
        "HDR Raw Image",
        "Demosaiced HDR RGB",
        "Normalized / Log Tone Mapped HDR",
        "Gamma Corrected",
        "White Balanced",
    ]
    images = [
        hdr_img_pillow,
        hdr_rgb_pillow,
        normalized_hdr_pillow,
        gamma_corr_pillow,
        white_balanced_pillow,
    ]

    # Font for titles
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Replace with a valid font path if necessary
    except IOError:
        font = ImageFont.load_default()

    # Draw images and titles on the canvas
    draw = ImageDraw.Draw(canvas)
    for i, (title, img) in enumerate(zip(titles, images)):
        col = i % grid_width
        row = i // grid_width
        x_offset = padding + col * (img_width + padding)
        y_offset = padding + row * (img_height + padding)

        # Paste the image
        canvas.paste(img.resize((img_width, img_height)), (x_offset, y_offset))

        # Add title text
        text_bbox = draw.textbbox((0, 0), title, font=font)  # Get bounding box for the title
        text_width = text_bbox[2] - text_bbox[0]  # Width of the title
        text_x = x_offset + img_width // 2 - text_width // 2  # Centered horizontally
        text_y = y_offset - 30  # Place title above the image
        draw.text((text_x, text_y), title, fill="black", font=font)

    canvas.save(output_file)
    print(f"Plot saved to {output_file}")

# output_range = 4
# input_intensity = 1/61 · (20·red + 40·green + blue)
# r, g, b = rgb / input_intensity
# log_base = bilat_filt(log(input_intensity))
# log_details = log(input_intensity) - log_base
# compression = log(output_range) / (max(log_base)-min(log_base))
# log_offset = -max(log_base) · compression
# output_intensity = exp(log_base · compression + log_offset + log_detail)
# rgb = r·output_intensity, g·output_intensity, b·output_intensity


def icam06(rgb_image, output_range=4):
    # Ensure the image is in float format for computations
    rgb_image = rgb_image.astype(np.float64)
    # Step 1 
    red, green, blue = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    input_intensity = (1 / 61) * (20 * red + 40 * green + blue)
    # Step 2 
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.divide(red, input_intensity, out=np.zeros_like(red), where=input_intensity > 0)
        g = np.divide(green, input_intensity, out=np.zeros_like(green), where=input_intensity > 0)
        b = np.divide(blue, input_intensity, out=np.zeros_like(blue), where=input_intensity > 0)
    # Step 3 
    log_input_intensity = np.log(input_intensity + 1e-8)  # Add a small value to prevent log(0)
    # Step 4
    log_base = cv2.bilateralFilter(log_input_intensity.astype(np.float32), 9, 50, 50)
    # Step 5 
    log_details = log_input_intensity - log_base
    # Step 6 
    compression = np.log(output_range) / (np.max(log_base) - np.min(log_base) + 1e-8)
    log_offset = -np.max(log_base) * compression
    # Step 7
    output_intensity = np.exp(log_base * compression + log_offset + log_details) / 1.5
    # Step 8
    tone_mapped_image = np.stack([r * output_intensity, g * output_intensity, b * output_intensity], axis=-1)
    # Clip to [0, 1] range
    tone_mapped_image = np.clip(tone_mapped_image, 0, 1)

    return tone_mapped_image
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

    
    path_6 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\06"
    filename_6 = [f"0{i}.CR3" if i < 10 else f"{i}.CR3" for i in range(0, 11)]
    
    # Load raw images
    raw_img_6 = load_data(path_6, filename_6)
    # Save the result
    plots_output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots'

    
    # Step 1: Calculate Exposure Times
    print("Calculating exposure times...")
    t0 = 1
    exposure_times = [t0 * (1/2)**n for n in range(len(filename_6))]
    print(f"Exposure times calculated: {exposure_times}")

    # Step 2: Perform HDR Combination
    print("Performing HDR combination of raw images...")
    hdr_img = hdr_combination(raw_img_6, exposure_times)
    # hdr_img = hdr(raw_img_6, threshold=0.8) 
    print("HDR combination complete.")

    # Step 3: Demosaic the HDR Raw Image
    print("Demosaicing the HDR raw image...")
    # hdr_rgb_image = cv2.cvtColor(hdr_img.astype(np.uint16), cv2.COLOR_BAYER_RG2RGB)
    hdr_rgb_image = demosaic(hdr_img)
    print("Demosaicing complete.")

    # Step 4: Logarithmic Tone Mapping
    print("Applying logarithmic tone mapping...")
    log_hdr = np.log(1 + hdr_rgb_image / np.max(hdr_rgb_image))  # Normalized for log mapping
    print("Logarithmic tone mapping complete.")

    # Step 5: Normalize HDR for Visualization
    print("Normalizing HDR for visualization...")
    normalized_hdr = (log_hdr / np.max(log_hdr) * 255).astype(np.uint8)
    print("Normalization complete.")

    # Step 6: Gamma Correction
    print("Applying gamma correction...")
    gamma_corr_img = gamma_correction(normalized_hdr / 255.0, gamma=0.3)
    gamma_corr_img_uint8 = (gamma_corr_img * 255).astype(np.uint8)  # Convert back to 8-bit
    print("Gamma correction complete.")

    # Step 7: White Balance
    print("Applying white balance...")
    white_balanced_img = white_balance(gamma_corr_img)
    print("White balance applied.")


    print("Icam06 Tone Mapping...")
    icam06_tone_mapped_img = icam06(white_balanced_img, output_range=4)
    print("Icam06 Tone Mapping complete.")


    # Step 8: Plotting HDR Processing Steps
    print("Plotting HDR processing steps...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # HDR Raw Image (Before Processing)
    axes[0, 0].imshow(hdr_img, cmap='gray')
    axes[0, 0].set_title('HDR Raw Image')
    axes[0, 0].axis('off')

    # Demosaiced RGB Image
    axes[0, 1].imshow(hdr_rgb_image.astype(np.uint8))
    axes[0, 1].set_title('Demosaiced HDR RGB')
    axes[0, 1].axis('off')

    # # Log Tone Mapped
    # axes[0, 2].imshow(log_hdr.astype(np.uint8))
    # axes[0, 2].set_title('Log Tone Mapped')
    # axes[0, 2].axis('off')

    # Normalized HDR
    axes[0, 2].imshow(normalized_hdr, cmap='gray')
    axes[0, 2].set_title('Normalized / Log Tone Mapped HDR')
    axes[0, 2].axis('off')

    # Gamma Corrected
    axes[1, 0].imshow(gamma_corr_img_uint8)
    axes[1, 0].set_title('Gamma Corrected')
    axes[1, 0].axis('off')

    # White Balanced
    axes[1, 1].imshow(white_balanced_img)
    axes[1, 1].set_title('White Balanced')
    axes[1, 1].axis('off')

    # White Balanced
    axes[1, 2].imshow(icam06_tone_mapped_img)
    axes[1, 2].set_title('ICAM06 Tone Mapped')
    axes[1, 2].axis('off')

    # Save the figure
    plt.tight_layout()
    output_file = os.path.join(plots_output_path, 'hdr_processing_steps.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

    output_file = os.path.join(plots_output_path, "hdr_processing_steps_pillow.png")
    create_hdr_plot_pillow(hdr_img, hdr_rgb_image, normalized_hdr, gamma_corr_img_uint8, white_balanced_img, output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()