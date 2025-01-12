import cv2
import numpy as np
from mtcnn import MTCNN


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=30, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # TODO: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            self.reference = self.detect_face(image)
            return self.reference

        # Size validation for input image
        if image is None or image.size == 0:
            print("Invalid input image")
            return None

        # Get the reference face rectangle and create a properly sized search window
        # Define the search window around the reference face.
        # [x, y, width, height] -> x and y are the top-left corner of the rectangle, width and height are the width and height of the rectangle
        # image.shape: The dimensions of the image as (height, width, channels)
        # The search window is defined by the reference face rectangle with an additional margin of 25 pixels
        # The origin (0, 0) is at the top-left corner - The y-axis increases as you move downward, and the x-axis increases as you move rightward
        ref_rect = self.reference["rect"]
        margin = self.tm_window_size
        search_top = max(ref_rect[1] - margin, 0)
        search_left = max(ref_rect[0] - margin, 0)
        search_bottom = min(ref_rect[1] + ref_rect[3] + margin, image.shape[0])
        search_right = min(ref_rect[0] + ref_rect[2] + margin, image.shape[1])
        search_window = image[search_top:search_bottom, search_left:search_right]

        # in self.reference["aligned"], size of template was 224x224 by using align_face method,
        # but the actual face in new frames is still its natural size (e.g., 140x110)
        # so we need to use the original size of the template for template matching
        # When doing template matching, OpenCV tries to find where the template (224x224) fits in the search window
        # our search window is based on the original face size plus margin
        # therefore we just crop the face at its original size, No resizing to 224x224
        # The template maintains the face's natural dimensions
        template = self.crop_face(self.reference["image"], self.reference["rect"])
        
        # Debug prints
        print(f"Search window size: {search_window.shape}")
        print(f"Template size: {template.shape}")

        # Validate search window size
        if (search_window.shape[0] < template.shape[0] or 
            search_window.shape[1] < template.shape[1]):
            print("Search window is smaller than the template .. Reinitializing tracker due to small search window ################")
            self.reference = self.detect_face(image)
            return self.reference

        result = cv2.matchTemplate(search_window, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        
        print(f"Template matching response: {min_val}")

        if min_val < self.tm_threshold:

            top_left = (min_loc[0] + search_left, min_loc[1] + search_top)
            face_rect = [top_left[0], top_left[1], ref_rect[2], ref_rect[3]]
            
            aligned = self.align_face(image, face_rect)
            
            return {
                "rect": face_rect,
                "image": image,
                "aligned": aligned,
                "response": min_val
            }

        print("Reinitializing tracker due to poor match ################")
        self.reference = self.detect_face(image)
        return self.reference


    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        if not (detections := self.detector.detect_faces(image)):
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
