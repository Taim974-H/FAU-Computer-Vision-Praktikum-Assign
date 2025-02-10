import cv2
import json
import matplotlib.pyplot as plt


json_path =  r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\results\train\53500107_d24b11b3c2_b_jpg.rf.6a78865b012c624e6caf705776d7b6b8_regions.json"
image_path = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\train\53500107_d24b11b3c2_b_jpg.rf.6a78865b012c624e6caf705776d7b6b8.jpg"  # Replace with your image file
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

# Load the region proposals from JSON
with open(json_path, "r") as f:  # Open the file properly
    regions = json.load(f)  # Now load works correctly

# Draw bounding boxes on the image
for region in regions:
    x, y, w, h = region["rect"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")  # Hide axes for better visualization
plt.show()
