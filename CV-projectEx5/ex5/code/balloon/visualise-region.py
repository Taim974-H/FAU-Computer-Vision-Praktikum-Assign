import cv2
import json
import matplotlib.pyplot as plt
import os

# Define file paths
base_dir = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\results\balloon_regions\training-examples\train"
image_dir = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\train"

file_name = "14666848163_8be8e37562_k_jpg.rf.0b34b472887a02ce894a77cc537d7026.jpg"

annotations_path = os.path.join(image_dir, "_annotations.coco.json")


# Paths to positive and negative samples
positive_json_path = os.path.join(base_dir, file_name, "positive_samples.json")
negative_json_path = os.path.join(base_dir, file_name, "negative_samples.json")
image_path = os.path.join(image_dir, file_name) 


# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Load COCO annotations
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Find the corresponding image ID in COCO dataset
image_id = None
for img in coco_data["images"]:
    if img["file_name"] == file_name:
        image_id = img["id"]
        break

if image_id is None:
    raise ValueError(f"Image {file_name} not found in {annotations_path}")

# Get ground truth bounding boxes for the image
ground_truth_regions = [
    ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id
]

# Function to load regions from JSON
def load_regions(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# Load positive and negative samples
positive_regions = load_regions(positive_json_path)
negative_regions = load_regions(negative_json_path)

print(f"Image Shape: {image.shape}")  # (Height, Width, Channels)


# # Draw ground truth bounding boxes (Blue)
# for x, y, w, h in ground_truth_regions:
#     cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)  # Blue


# Draw positive samples (Green)
for region in positive_regions:
    x, y, w, h = region["rect"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green

# # Draw negative samples (Red)
# for region in negative_regions:
#     x, y, w, h = region["rect"]
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")  # Hide axes
plt.title("Green: Positive Samples | Red: Negative Samples")
plt.show()

