import pickle

import numpy as np
import sys
from pathlib import Path

# Add 'src' directory (parent of 'cvproj_exc4') to Python's search path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cvproj_exc4.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    
    def run(self):
        # Train the Classifier with training set
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # Predict labels and similarities for test set
        pred_y, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # Calculate identification rates for different similarity thresholds
        identification_rates = []
        similarity_thresholds = []

        for false_alarm_rate in self.false_alarm_rate_range:
            # Get threshold for current false alarm rate
            # False Alarm Rate (FAR) is based on unknown subjects being incorrectly accepted
            similarity_threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            similarity_thresholds.append(similarity_threshold)

            # Apply threshold to predictions
            corrected_pred = np.where(
                similarities >= similarity_threshold,
                pred_y,
                UNKNOWN_LABEL
            )

            # Calculate identification rate with corrected predictions
            # Detection and Identification Rate (DIR) is based on known subjects being correctly identified
            identification_rate = self.calc_identification_rate(corrected_pred)
            identification_rates.append(identification_rate)

        return {
            "false_alarm_rates": self.false_alarm_rate_range,
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }

    def select_similarity_threshold(self, similarities, false_alarm_rate):

        # Get similarities only for unknown subjects in test set
        unknown_mask = np.array(self.test_labels) == UNKNOWN_LABEL
        unknown_similarities = similarities[unknown_mask]
        
        if len(unknown_similarities) == 0:
            return np.min(similarities)  # fallback if no unknown subjects
        

        # Lower threshold (e.g. 0.5):
        # - More unknown people get scores above it
        # - Higher false alarm rate (maybe 50%)

        # Higher threshold (e.g. 0.9):
        # - Fewer unknown people get scores above it
        # - Lower false alarm rate (maybe 5%)
        percentile = (1 - false_alarm_rate) * 100
        threshold = np.percentile(unknown_similarities, percentile)
        # this should gives us a threshold where percentile% of unknown people would be wrongly accepted
        return threshold

    def calc_identification_rate(self, prediction_labels):

        ground_truth = np.array(self.test_labels)
        predictions = np.array(prediction_labels)
        
        # Only consider known subjects
        known_mask = ground_truth != UNKNOWN_LABEL
        
        if not np.any(known_mask):
            return 0.0  # Return 0 if no known subjects
        
        # Calculate accuracy for known subjects
        known_correct = np.logical_and(
            known_mask,
            ground_truth == predictions
        )
        
        return float(np.sum(known_correct)) / np.sum(known_mask)
    
###################################

# similarities: "Are these faces similar enough to be considered the same person?"
# Similarity Score: 0.8 (for example)
# If threshold = 0.7: Accept as match (0.8 ≥ 0.7)
# If threshold = 0.9: Reject as match (0.8 < 0.9)

# LOW threshold:

# More matches accepted (both correct and incorrect)
# Higher false alarm rate (accepting unknown people)
# Higher identification rate (correctly identifying known people)


# HIGH threshold:

# Fewer matches accepted
# Lower false alarm rate (rejecting unknown people)
# Lower identification rate (might reject known people)

# example
# Unknown people's similarity scores: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# If we want 10% false alarm rate:
# - We need a threshold where only 10% of unknown people get scores above it
# - percentile = (1 - 0.1) * 100 = 90
# - This means we want the threshold where 90% of scores are below it
# - If we set threshold = 0.85, then only one score (1.0) is above it
# - This gives us roughly our desired 10% false alarm rate

###################################

# We have:
# - test_labels (containing both known and unknown subjects)
# - test_embeddings (face features for all test subjects)
# - false_alarm_rate_range (e.g., [0.001, 0.002, ..., 1.0])
# - trained classifier that gives us predictions and similarity scores

# For each false alarm rate in our range, we:
# a) First get similarity scores and initial predictions
# b) Then find the threshold that would give us that false alarm rate
#     # Look at similarities of UNKNOWN subjects only..
#     # If we want FAR = 0.1 (10%):
#     # We need a threshold where 10% of unknown subjects get wrongly accepted
#     # So we find the similarity value where 90% of unknown similarities fall below it
# c) Use this threshold to correct our predictions
#     # If similarity < threshold: predict as UNKNOWN
#     # If similarity >= threshold: keep original prediction
# d) Calculate identification rate using corrected predictions

# # For each false_alarm_rate, we get:
# - A threshold that would give us that FAR
# - An identification rate (DIR) achieved at that threshold

# # This creates our curve:
# X-axis: false alarm rates we tested
# Y-axis: identification rates we achieved at each FAR
