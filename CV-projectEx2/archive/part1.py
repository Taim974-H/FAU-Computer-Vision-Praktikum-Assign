import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import convolve
import os

# Load raw image data (Bayer pattern)
save_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\plots\\'

raw = rawpy.imread(r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\03\IMG_4782.CR3')
raw_img = np.array(raw.raw_image_visible)  # shape = (4660, 6984)

# Display the Bayer pattern as a grayscale image
plt.imshow(raw_img, cmap='gray')
plt.title("Bayer Pattern Data")
plt.colorbar()
plt.savefig(os.path.join(save_path, 'raw_img.png'))
plt.close()

# Convert to RGB using `rawpy`'s default demosaicing for comparison
rgb_image = raw.postprocess()
plt.imshow(rgb_image)
plt.title("Demosaiced RGB Image")
plt.savefig(os.path.join(save_path, 'rgb_image.png'))
plt.close()

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
reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]
reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]
reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]

# Display the reconstructed image
plt.imshow(reconstructed_img.astype(np.uint8))
plt.title("Reconstructed RGB Image via Interpolation")
plt.colorbar()
plt.savefig(os.path.join(save_path, 'reconstructed_img.png'))
plt.close()


section_reconst_img = reconstructed_img[:10,:10]
plt.imshow(section_reconst_img.astype(np.uint8))
plt.title("section_reconst_img RGB Image via Interpolation")
plt.colorbar()
plt.savefig(os.path.join(save_path, 'section_reconst_img.png'))
plt.close()

#----------------------------------------------------------------------------------------------------------------
#Gamma Correction
#Normalize the data
a = np.percentile(reconstructed_img,0.01)
b = np.percentile(reconstructed_img,99.99)

normalized_data = (reconstructed_img-a)/(b-a) #shape = (4660, 6984, 3)

#Clip the values to [0, 1]
normalized_data[normalized_data < 0] = 0
normalized_data[normalized_data > 1] = 1

#applying gamma correction
gamma = 0.3
normalized_gamma_corrected = normalized_data**gamma

#scale back to original data
# final_gamma_corrected = normalized_gamma_corrected * (b - a ) + a

#Plot the gamma-corrected image
plt.figure(figsize=(8, 8))
plt.imshow(normalized_gamma_corrected)
plt.title("Gamma Corrected Image")
plt.axis("off")
plt.savefig(os.path.join(save_path, 'Gamma_Corrected_Image.png'))
plt.close()

#-------------------------------------------------------------------------------------------------------------------------
#White Balance
#calculate the mean of color channels
mean_red = np.mean(normalized_gamma_corrected[:,:,0])
mean_green = np.mean(normalized_gamma_corrected[:,:,1])
mean_blue = np.mean(normalized_gamma_corrected[:,:,2])

# Scaling Factor for Each Channel
scale_red = mean_green/mean_red
scale_green = mean_green/mean_green
scale_blue = mean_green/mean_blue

#equalize the intensities
white_balanced_img = normalized_gamma_corrected.copy()
white_balanced_img[:,:,0] = (white_balanced_img[:,:,0]) * scale_red
white_balanced_img[:,:,1] = (white_balanced_img[:,:,1]) * scale_green
white_balanced_img[:,:,2] = (white_balanced_img[:,:,2]) * scale_blue

# Clip the values to avoid overflow
white_balanced_img = np.clip(white_balanced_img, 0, 1)

# Scale back to [0, 255] for proper display
white_balanced_img_display = (white_balanced_img * 255).astype(np.uint8)

# Display the white-balanced image
plt.figure(figsize=(8, 8))
plt.imshow(white_balanced_img_display)
plt.title("White Balanced Image (Gray World)")
plt.axis("off")
plt.savefig(os.path.join(save_path, 'white_balanced_Image_display.png'))
plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
#5 Show that Sensor Data is Linear
path = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx2\exercise_2_data\exercise_2_data\02"
filename = [f"IMG_304{i}.CR3" for i in range(4,10)]

raw_files = []

for name in filename:
    file_path = os.path.join(path,name)
    raw = rawpy.imread(file_path)
    raw_img = np.array(raw.raw_image_visible)
    raw_files.append(raw_img)

print(0)