### Summary of Processing Steps for part 8 (win):
1. **Load raw image**: Read the raw sensor data.
2. **Demosaicing**: Convert raw Bayer data into a full-color image.
3. **Normalize**: Scale image to 0-255 for proper display.
4. **Enhance contrast**: Apply CLAHE for local contrast improvement.
5. **Enhance color**: Boost saturation and brightness in HSV space.
6. **Sharpen image**: Apply sharpening filter to improve details.
7. **Reduce noise**: Apply denoising to smooth the image.
8. **Save and display**: Save the final image and display histograms for analysis.
9. **Error handling**: Catch any issues during processing and report them.


### 1. **Loading Raw Image**
   - **Method**: `rawpy.imread(input_path)`
   - **Purpose**: Loads a raw image file (e.g., CR3 format) into memory for processing. This method reads the raw sensor data without applying any processing like demosaicing or color adjustments.

### 2. **Demosaicing (Conversion from Bayer Pattern to RGB)**
   - **Method**: `raw.postprocess()`
   - **Purpose**: Converts the raw Bayer pattern data into a full-color RGB image using the **AHD (Adaptive Homogeneity-Directed)** demosaicing algorithm. It also applies some initial corrections such as white balance (using camera settings) and brightness adjustments.

### 3. **Normalization**
   - **Method**: `cv2.normalize()`
   - **Purpose**: Scales the pixel values of the image from their original range (based on 16-bit depth) to the 0-255 range (for 8-bit image formats) for proper display and saving.

### 4. **Contrast Enhancement (CLAHE)**
   - **Method**: `cv2.createCLAHE()`, `clahe.apply()`
   - **Purpose**: Improves local contrast in the image by enhancing the lightness channel (L) in the LAB color space using **CLAHE** (Contrast Limited Adaptive Histogram Equalization). This helps bring out details in areas that are too dark or too bright.

### 5. **Color Enhancement**
   - **Method**: `cv2.cvtColor()`, `cv2.multiply()`
   - **Purpose**: Modifies the **saturation** and **brightness** in the HSV color space to make the colors more vivid and to brighten the overall image slightly. Saturation is boosted by multiplying the saturation channel, and the brightness (value channel) is also slightly increased.

### 6. **Image Sharpening**
   - **Method**: `cv2.filter2D()`
   - **Purpose**: Applies a custom **sharpening filter** using a convolution kernel to enhance the edges and details in the image, making it appear crisper and more defined.

### 7. **Noise Reduction**
   - **Method**: `cv2.fastNlMeansDenoisingColored()`
   - **Purpose**: Reduces noise from the image using **Non-Local Means Denoising**, which smooths the image while preserving edges and fine details. It applies noise reduction in both the color and brightness channels.

### 8. **Saving and Displaying the Final Image**
   - **Method**: `cv2.imwrite()`, `plt.imshow()`
   - **Purpose**: The processed image is saved to the specified output path, and also displayed using **matplotlib**. The histograms of the color channels (red, green, blue) are plotted to visualize the pixel distribution before saving the final result.

### 9. **Error Handling**
   - **Method**: `try-except`
   - **Purpose**: Ensures that any errors that occur during processing (such as file read/write issues or invalid image formats) are caught and reported without crashing the program.

---