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

#############################################################

# Demosaicing ------------------------------------------------

#############################################################


def demosaic(raw_img):
    # Padding to handle edge cases
    padded_img = np.pad(raw_img, pad_width=1, mode='symmetric')
    height, width = padded_img.shape

    # Initialize the reconstructed image
    reconstructed_img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))

    # Create Bayer masks for BGGR pattern
    red_mask = np.zeros_like(padded_img)
    green_mask = np.zeros_like(padded_img)
    blue_mask = np.zeros_like(padded_img)

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

    # Convolution kernels for interpolation
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float64) / 4.0

    # Interpolate each channel
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
    reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]  # Red
    reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]  # Green
    reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]  # Blue

    return reconstructed_img


#############################################################

# Gamme Correction ------------------------------------------

#############################################################

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


#############################################################

# White balancing ------------------------------------------

#############################################################
    
    
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

    return white_balanced_img


#############################################################

# HDR Combination ------------------------------------------

#############################################################


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

# Log Tone Mapping ------------------------------------------

#############################################################

def tone_map_log(hdr_image):    
    hdr_image = hdr_image.astype(np.float32)
    
    log_hdr = np.log(hdr_image + 1e-8)  # Small epsilon to avoid log(0)
    
    log_hdr -= log_hdr.min()
    log_hdr /= log_hdr.max()
    
    return log_hdr  # Return normalized float32 image

#############################################################

# ICAM06 ----------------------------------------------------

#############################################################


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
    log_base = cv2.bilateralFilter(log_input_intensity.astype(np.float32), 5, 0.5,5.0)
    # Step 5 
    log_details = log_input_intensity - log_base
    # Step 6 
    compression = np.log(output_range) / (np.max(log_base) - np.min(log_base) + 1e-8)
    log_offset = -np.max(log_base) * compression
    # Step 7
    output_intensity = np.exp(log_base * compression + log_offset + log_details)
    # Step 8
    tone_mapped_image = np.stack([r * output_intensity, g * output_intensity, b * output_intensity], axis=-1)
    # Clip to [0, 1] range
    tone_mapped_image = np.clip(tone_mapped_image, 0, 1)

    return tone_mapped_image

#############################################################

# Using Pillow Lib for HDR Process Plots --------------------

#############################################################


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




#############################################################

# Main function ---------------------------------------------

#############################################################

def main():
    # Update plots output path to your Windows path
    plots_output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots'

    #############################################################

    # Section 2: Demosaicing
    # Section 3: Improve Luminosity
    # Section 4: White Balance

    #############################################################


    # Demosaicing
    path_2 = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\03"
    filename_2 = ["IMG_4782.CR3"]
    raw_img_2 = load_data(path_2, filename_2)
    
    # Demosaic IMG_4782
    reconstructed_img = demosaic(raw_img_2[0])

    reconstructed_img -= reconstructed_img.min()
    reconstructed_img /= reconstructed_img.max()

    # Improve Luminosity
    luminosity_corr_img = gamma_correction(reconstructed_img, gamma=0.5)

    # White Balance
    white_balance_img = white_balance(luminosity_corr_img)
    white_balance_display = (white_balance_img * 255).astype(np.uint8)

    # Visualize reconstructed image
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed RGB Image via Interpolation")
    plt.colorbar()
    plt.savefig(os.path.join(plots_output_path, 'Bird_Demosaic_reconstructed_img.png'))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(luminosity_corr_img)
    plt.title("Gamma Corrected Image")
    plt.axis("off")
    plt.savefig(os.path.join(plots_output_path, 'Bird_Gamma_Corrected_Image.png'))
    plt.close()    
    
    plt.figure(figsize=(8, 8))
    plt.imshow(white_balance_display)
    plt.title("White Balanced Image")
    plt.axis("off")
    plt.savefig(os.path.join(plots_output_path, 'Bird_White_Balanced_Image.png'))
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
    # Section 7: ICAM06 Implementation

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
    hdr_rgb_image -= hdr_rgb_image.min()
    hdr_rgb_image /= hdr_rgb_image.max()
    print("Demosaicing complete.")    

    # Step 4: Logarithmic Tone Mapping
    print("Applying logarithmic tone mapping...")
    log_hdr = tone_map_log(hdr_rgb_image)
    # log_hdr = np.log(1 + hdr_rgb_image / np.max(hdr_rgb_image))  # Normalized for log mapping
    print("Logarithmic tone mapping complete.")

    # Step 5: Normalize HDR for Visualization
    print("Normalizing HDR for visualization...")
    normalized_hdr = (log_hdr * 255).astype(np.uint8)
    print("Normalization complete.")

    # Step 6: Gamma Correction
    print("Applying gamma correction...")
    gamma_corr_img = gamma_correction(log_hdr, gamma=0.5)
    print("Gamma correction complete.")
    
    # Step 7: White Balance
    print("Applying white balance...")
    white_balanced_img = white_balance(gamma_corr_img)
    white_balance_img_display = (white_balanced_img * 255).astype(np.uint8)
    print("White balance applied.")

    # Step 8: ICAM06 Tone Mapping
    print("Icam06 Tone Mapping...")
    icam06_tone_mapped_img = icam06(white_balanced_img, output_range=8)
    print("Icam06 Tone Mapping complete.")


    # Step 9: Plotting HDR Processing Steps
    print("Plotting HDR processing steps...")

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot titles and images
    plots = [
        ('HDR Raw Image', hdr_img, 'gray'),
        ('Demosaiced HDR RGB', hdr_rgb_image, None),
        ('Normalized HDR after Log Tone Mapped', normalized_hdr, None),
        ('Gamma Corrected', gamma_corr_img, None),
        ('White Balanced', white_balance_img_display, None),
        ('ICAM06 Tone Mapped', icam06_tone_mapped_img, None)
    ]

    # Directory to save individual plots
    individual_plots_dir = os.path.join(plots_output_path, 'individual_plots_part06')
    os.makedirs(individual_plots_dir, exist_ok=True)

    # Plotting each step
    for idx, (title, img, cmap) in enumerate(plots):
        row, col = divmod(idx, 3)
        axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

        # Save individual plot
        individual_file = os.path.join(individual_plots_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(individual_file)
        plt.close()
        print(f"Individual plot saved to {individual_file}")

    # Save the combined figure
    plt.tight_layout()
    output_file = os.path.join(plots_output_path, 'hdr_processing_steps.png')
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")
    plt.close()

    # output_file = os.path.join(plots_output_path, "hdr_processing_steps_pillow.png")
    # create_hdr_plot_pillow(hdr_img, hdr_rgb_image, normalized_hdr, gamma_corr_img_uint8, white_balanced_img, output_file)
    # print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()