import os
import pickle

import cv2
import numpy as np
from pathlib import Path
import sys

# Add 'src' directory (parent of 'cvproj_exc4') to Python's search path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cvproj_exc4.config import Config


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(Config.resnet50)

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# # The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:
    def __init__(self, num_neighbours=5, max_distance=0.4, min_prob=0.85):
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Load face recognizer from pickle file if available
        if os.path.exists(Config.rec_gallery):
            self.load()

    def save(self):
        with open(Config.rec_gallery, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    def load(self):
        with open(Config.rec_gallery, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    def partial_fit(self, face, label):
        # Get color embedding
        color_face = face.copy()
        color_embedding = self.facenet.predict(color_face)
        color_embedding = color_embedding / np.linalg.norm(color_embedding)
        
        # Get grayscale embedding
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        gray_embedding = self.facenet.predict(gray_face)
        gray_embedding = gray_embedding / np.linalg.norm(gray_embedding)
        
        # Store both embeddings with proper normalization
        self.embeddings = np.vstack((self.embeddings, color_embedding, gray_embedding))
        self.labels.extend([label, label])  # Add label twice for color and gray
        
        return color_embedding, gray_embedding, label

    def predict(self, face):
        if len(self.labels) == 0:
            return "unknown", 0.0, float('inf')

        # Get embeddings for query face
        color_embedding = self.facenet.predict(face)
        color_embedding = color_embedding / np.linalg.norm(color_embedding)
        
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        gray_embedding = self.facenet.predict(gray_face)
        gray_embedding = gray_embedding / np.linalg.norm(gray_embedding)

        # Calculate distances using cosine similarity
        color_distances = 1 - np.dot(self.embeddings[::2], color_embedding.T).flatten()
        gray_distances = 1 - np.dot(self.embeddings[1::2], gray_embedding.T).flatten()
        
        # Weighted combination of distances
        combined_distances = 0.6 * color_distances + 0.4 * gray_distances

        # Find k nearest neighbors
        k = min(self.num_neighbours, len(self.labels) // 2)
        nearest_indices = np.argpartition(combined_distances, k)[:k]
        nearest_distances = combined_distances[nearest_indices]
        nearest_labels = [self.labels[i * 2] for i in nearest_indices]

        # Calculate probability and predicted label
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        max_count_idx = np.argmax(counts)
        predicted_label = unique_labels[max_count_idx]
        probability = counts[max_count_idx] / k

        # Calculate minimum and average distance to predicted class
        class_distances = nearest_distances[np.array(nearest_labels) == predicted_label]
        min_class_distance = np.min(class_distances) if len(class_distances) > 0 else float('inf')
        avg_class_distance = np.mean(class_distances) if len(class_distances) > 0 else float('inf')

        # More stringent unknown detection
        is_unknown = False
        
        # Check if distances are too spread out (indicates uncertainty)
        distance_spread = np.std(class_distances) if len(class_distances) > 1 else float('inf')
        if distance_spread > 0.15:  # High variance in distances suggests uncertainty
            is_unknown = True
            
        # Check if average distance is too high
        if avg_class_distance > self.max_distance * 0.8:  # Using 80% of max_distance for average
            is_unknown = True
            
        # Check if minimum distance is too high
        if min_class_distance > self.max_distance:
            is_unknown = True
            
        # Check if probability is too low
        if probability < self.min_prob:
            is_unknown = True

        # Check if there's too much competition from other classes
        if len(unique_labels) > 1:
            second_best_count = np.sort(counts)[-2]
            if counts[max_count_idx] - second_best_count <= 1:  # Too close between top classes
                is_unknown = True

        if is_unknown:
            return "unknown", probability, min_class_distance

        return predicted_label, probability, min_class_distance


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=25):
        # TODO: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # For tracking convergence
        self.objective_values = []

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.cluster_gallery):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open(Config.cluster_gallery, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        with open(Config.cluster_gallery, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # TODO
    def partial_fit(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack((self.embeddings, embedding))
        return embedding

    # TODO
    def fit(self):
        if len(self.embeddings) < self.num_clusters:
            raise ValueError("Number of samples less than number of clusters")
        
        # Randomly initialize cluster centers 
        # Number of cluster centers is determined by num_clusters
        random_indices = np.random.choice(
            len(self.embeddings), # number of samples
            size=self.num_clusters, 
            replace=False # this just means that the same index can't be chosen twice
        )
        self.cluster_center = self.embeddings[random_indices] # cluster centers are the embeddings at the random indices

        prev_objective = float('inf')
        self.objective_values = []

        # K-means iteration
        for _ in range(self.max_iter):
            # Assignment step: assign each point to nearest cluster
            distances = np.zeros((len(self.embeddings), self.num_clusters)) # initialize distances matrix
            for i in range(self.num_clusters):
                distances[:, i] = np.linalg.norm(
                    self.embeddings - self.cluster_center[i], # calculate distance between each embedding and cluster center
                    axis=1
                )
            
            self.cluster_membership = np.argmin(distances, axis=1) # assign each embedding to the nearest cluster
            
            # Update step: recalculate cluster centers
            new_centers = np.zeros_like(self.cluster_center) # initialize new cluster centers
            for i in range(self.num_clusters):
                cluster_points = self.embeddings[self.cluster_membership == i] # get all embeddings in the cluster
                if len(cluster_points) > 0: 
                    new_centers[i] = np.mean(cluster_points, axis=0) # calculate new cluster center as the mean of all embeddings in the cluster
                else:
                    # If a cluster is empty, reinitialize its center randomly
                    new_centers[i] = self.embeddings[
                        np.random.randint(len(self.embeddings))
                    ]
            
            # Calculate objective function (sum of squared distances)
            objective = 0
            for i in range(self.num_clusters):
                cluster_points = self.embeddings[self.cluster_membership == i]
                if len(cluster_points) > 0:
                    objective += np.sum(
                        np.linalg.norm(
                            cluster_points - self.cluster_center[i], 
                            axis=1
                        ) ** 2
                    )
            
            self.objective_values.append(objective)
            
            # Check convergence
            # this is just a simple check to see if the objective function is not changing much
            if abs(prev_objective - objective) < 1e-6: 
                break
                
            self.cluster_center = new_centers
            prev_objective = objective
        
        return self.cluster_membership, self.objective_values


    # TODO
    def predict(self, face):

        embedding = self.facenet.predict(face)
        
        # Calculate distances to all cluster centers
        distances = np.array([
            np.linalg.norm(embedding - center) 
            for center in self.cluster_center
        ])
        
        best_cluster = np.argmin(distances) # best matching cluster
        
        return best_cluster, distances

# Reference
# stat quest k-means clustering video: https://www.youtube.com/watch?v=4b5d3muPQmA