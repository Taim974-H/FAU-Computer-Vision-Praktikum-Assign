import cv2
import json
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Define file paths
base_dir = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\code\results\balloon_regions\training-examples\train"
image_dir = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx5\ex5\data\balloon_dataset\train"
annotations_path = os.path.join(image_dir, "_annotations.coco.json")

# Load COCO annotations
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Get all image IDs
image_ids = [img["id"] for img in coco_data["images"]]
current_index = 0  # Index to track current image

# Create Tkinter window
root = tk.Tk()
root.title("Bounding Box Viewer")

# Canvas for displaying images
label = Label(root)
label.pack()

def load_regions(json_path):
    """Load regions from JSON file if it exists."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return []  # Return empty list if file is missing

def display_image():
    """Load image & draw bounding boxes."""
    global current_index, label

    # Get current image ID and filename
    image_id = image_ids[current_index]
    image_info = next((img for img in coco_data["images"] if img["id"] == image_id), None)

    if image_info is None:
        return
    
    file_name = image_info["file_name"]
    image_path = os.path.join(image_dir, file_name)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Get ground truth bounding boxes
    ground_truth_regions = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id]

    # Load positive/negative samples
    positive_json_path = os.path.join(base_dir, file_name, "positive_samples.json")
    negative_json_path = os.path.join(base_dir, file_name, "negative_samples.json")

    positive_regions = load_regions(positive_json_path)
    negative_regions = load_regions(negative_json_path)

    # Draw ground truth bounding boxes (Blue)
    for x, y, w, h in ground_truth_regions:
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)

    # Draw positive samples (Green)
    for region in positive_regions:
        x, y, w, h = region["rect"]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # # Draw negative samples (Red)
    # for region in negative_regions:
    #     x, y, w, h = region["rect"]
    #     cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    # Convert image to PIL format
    image = Image.fromarray(image)
    image = image.resize((800, 600))  # Resize for display
    img_tk = ImageTk.PhotoImage(image)

    # Update Tkinter label
    label.config(image=img_tk)
    label.image = img_tk

    root.title(f"Image ID: {image_id} | File: {file_name}")

def next_image():
    """Go to next image."""
    global current_index
    if current_index < len(image_ids) - 1:
        current_index += 1
        display_image()

def prev_image():
    """Go to previous image."""
    global current_index
    if current_index > 0:
        current_index -= 1
        display_image()

# Buttons for navigation
prev_button = Button(root, text="⬅️ Previous", command=prev_image)
prev_button.pack(side="left", padx=20, pady=10)

next_button = Button(root, text="Next ➡️", command=next_image)
next_button.pack(side="right", padx=20, pady=10)

# Initial display
display_image()

# Start Tkinter event loop
root.mainloop()
