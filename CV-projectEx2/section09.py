import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from exif import Image as ExifImage

exposure_times = []

def get_exposure_time(image_path):
    """Extracts the exposure time from an image's EXIF metadata using the exif module."""
    try:
        with open(image_path, 'rb') as img_file:
            img_exif = ExifImage(img_file)
            if img_exif.has_exif and hasattr(img_exif, 'exposure_time'):
                return img_exif.exposure_time  # returns a float (or a Ratio that converts to float)
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
    return None

def load_data(path, filenames, resize_dim=None):
    global exposure_times
    images = []
    exposure_times.clear()  # reset global exposures

    for name in filenames:
        img_path = os.path.join(path, name)
        # Load image with cv2
        # convert to RGB
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            print(f"Warning: Unable to load {img_path}")
            continue
        
        if resize_dim is not None:
            img = cv2.resize(img, resize_dim)
        
        images.append(img)
        
        exp = get_exposure_time(img_path)
        if exp is not None:
            exposure_times.append(exp)
        else:
            print(f"Warning: Exposure time not found for {img_path}.")
    
    return images

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
    
def inverse_gamma_correction(rgb_img, gamma):

    img = normalize_0_to_1(rgb_img)
    linearized_img = img ** (1/gamma) # Apply inverse gamma correction
    return linearized_img


def estimate_gamma(images, exposures):
    """
    - Estimate gamma using the relationship:
         I = k * (exposure)^(1/gamma)
    
    - Take the natural logarithm:
         log(I) = log(k) + (1/gamma)*log(exposure)
    
    - The slope of the log-log regression is 1/gamma so that:
         gamma_est = 1 / slope
    """
    avg_intensities = []
    # Use the original 8-bit images for intensity estimation
    for img in images:
        # Convert to float in [0,1] and compute overall average
        norm_img = np.asarray(img, dtype=np.float32) / 255.0
        avg_intensity = np.mean(norm_img)
        avg_intensities.append(avg_intensity)
    avg_intensities = np.array(avg_intensities)
    exposures = np.array(exposures, dtype=np.float64)
    
    log_exposures = np.log(exposures)
    log_intensities = np.log(avg_intensities)
    
    slope, intercept = np.polyfit(log_exposures, log_intensities, 1)
    gamma_est = 1.0 / slope
    plt.figure(figsize=(8,6))
    plt.scatter(log_exposures, log_intensities, label="Data Points")
    plt.plot(log_exposures, slope*log_exposures + intercept, 'r-', label=f"Fit (gamma={gamma_est:.2f})")
    plt.xlabel("log(Exposure Time)")
    plt.ylabel("log(Avg Intensity)")
    plt.title("Gamma Estimation")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return gamma_est


# def plot_gamma_curves(output_file,gamma):
#     x = np.linspace(0, 1, 256)
#     y_gamma = np.power(x, gamma)        # forward gamma correction curve
#     y_inverse = np.power(x, 1.0/gamma)    # inverse gamma correction curve

#     plt.figure(figsize=(8, 6))
#     plt.plot(x, x, 'k--', label="Linear (no gamma)")
#     plt.plot(x, y_gamma, 'b-', label=f"Gamma Correction (x^({gamma:.2f}))")
#     plt.plot(x, y_inverse, 'r-', label=f"Inverse Gamma Correction (x^(1/{gamma:.2f}))")
#     plt.xlabel("Input Intensity (normalized)")
#     plt.ylabel("Output Intensity (normalized)")
#     plt.title("Gamma Correction Curves")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(output_file)
#     plt.close()
#     print(f"Gamma curves plot saved successfully")


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
    raw_images = np.array(raw_images)
    exposure_times = np.array(exposure_times)
    
    # Weight function: give more weight to middle-range pixel values
    def weight_function(z):
        return 1.0 - np.abs(z - 0.5) * 2.0
    
    # Initialize HDR image as zero
    hdr_image = np.zeros_like(raw_images[0], dtype=np.float64)
    
    for img, exposure in zip(raw_images, exposure_times):
        img_float = img.astype(np.float64)
        weights = weight_function(img_float / np.max(img_float))
        hdr_image += weights * img_float / exposure
    
    return hdr_image

#############################################################

# Main function ---------------------------------------------

#############################################################

def main():

    path_additional = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_additional_data\additional-exercise"
    filename_additional = [f"A45A69{i}.JPG" for i in range(23, 35)]

    # resizing needed to fix memory issues
    resize_dim = (800, 600)
    jpg_img_additional = load_data(path_additional, filename_additional, resize_dim=resize_dim)
    # Save the result
    output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots\additional-exercise'
    os.makedirs(output_path, exist_ok=True)

    #############################################################
    image = np.array(jpg_img_additional)
    image = np.asarray(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # estimate gamma using the original images (before any correction)
    gamma_estimated = estimate_gamma(jpg_img_additional, exposure_times)
    print(f"Estimated gamma: {gamma_estimated:.2f}")

    # plot_gamma_curves(gamma=gamma_estimated, output_file=os.path.join(output_path, "gamma_curves.png"))


    linearized_img = inverse_gamma_correction(image, gamma_estimated)
    white_balanced_hdr = white_balance(linearized_img)
    hdr_image = hdr_combination(white_balanced_hdr, exposure_times)

    # logarithmic tone mapping
    hdr_log = np.log1p(hdr_image)  # log1p computes log(1 + hdr_image)

    # normalize the log-mapped image to [0,255] (min-max normalization)
    hdr_log_norm = (hdr_log - np.min(hdr_log)) / (np.max(hdr_log) - np.min(hdr_log)) * 255.0 # (x - min) / (max - min) * 255
    hdr_log_norm = np.clip(hdr_log_norm, 0, 255).astype(np.uint8)

    #plot hdr image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(hdr_log_norm, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("JPG to HDR Image")
    plt.show()
    plt.close()

    cv2.imwrite(os.path.join(output_path, "jpg_to_hdr_additional.jpg"), hdr_log_norm )
    print("HDR Image saved successfully.")
        

if __name__ == "__main__":
    main()