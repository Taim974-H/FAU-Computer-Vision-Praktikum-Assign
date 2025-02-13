import numpy as np
import os
import cv2

def load_data(path, filenames):
    images = []
    for name in filenames:
        img_path = os.path.join(path, name)
        img = cv2.imread(img_path) 
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Unable to load {img_path}")
    return images

def inverse_gamma_correction(rgb_img, gamma=2.2):
    return np.power(rgb_img, gamma)


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
    white_balanced_img = np.clip(white_balanced_img, 0, 255)

    return white_balanced_img

def hdr_combination(raw_images, exposure_times):

    raw_images = np.array(raw_images)
    exposure_times = np.array(exposure_times)


    def weight_function(z):
        return 1.0 - np.abs(z - 0.5) * 2.0
    
    hdr_image = np.zeros_like(raw_images[0], dtype=np.float64)
    for img, exposure in zip(raw_images, exposure_times):
        img_float = img.astype(np.float64)
        weights = weight_function(img_float / np.max(img_float))
        hdr_image += weights * img_float / exposure
    return hdr_image

def main():
    # Load images
    path_additional = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_additional_data\additional-exercise"
    filename_additional = [f"A45A69{i}.JPG" for i in range(23, 35)]
    jpg_img_additional = load_data(path_additional, filename_additional)
    exposure_times = np.array([13, 6, 3.2, 1.6, 0.8, 1/2, 1/5, 1/10, 1/20, 1/40, 1/80, 1/160])

    # normalize
    images = np.array(jpg_img_additional, dtype=np.float32) / 255.0

    # Linearize pixel values, apply white balance, and combine images
    linearized_images = [inverse_gamma_correction(img, gamma=0.3) for img in images]
    white_balanced_images = [white_balance(img) for img in linearized_images]
    hdr_image = hdr_combination(white_balanced_images, exposure_times)

    # Save the HDR image
    output_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots\additional-exercise'
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, "hdr_image.png"), hdr_image)

    print("HDR Image saved successfully.")

if __name__ == "__main__":
    main()