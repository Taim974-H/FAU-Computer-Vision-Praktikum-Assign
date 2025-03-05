import sys
import logging
from pathlib import Path
from collections.abc import Callable
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cvproj_exc4.config import Config 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.calibration import CalibratedClassifierCV
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress(message: str, percent: int):
    """Prints a progress message with percentage."""
    logging.info(f"{message} [{percent}%]")

def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    try:
        # Preprocessing
        preprocessor = make_pipeline(
            RobustScaler(),
            Normalizer(norm='l2'),
            Nystroem(kernel='rbf', random_state=42, n_components=100)
            )
        x_train_processed = preprocessor.fit_transform(x_train)

        # remap unknowns (-1) to a new pseudo-class ( we do this to avoid class imbalance)
        known_mask = (y_train != -1)
        max_known = int(np.max(y_train[known_mask])) if np.any(known_mask) else 0
        new_y_train = np.copy(y_train)
        new_y_train[~known_mask] = max_known + 1  # pseudo-class

        # Train base classifier (Linear SVC as using rbf was not time efficient) and calibrate using isotonic regression.
        # base_clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        base_clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000)
        grid_search = GridSearchCV(base_clf, param_grid, cv=2, scoring='balanced_accuracy')
        grid_search.fit(x_train_processed, new_y_train)
        best_base_clf = grid_search.best_estimator_

        # Calibrate classifier using cv="prefit" - prefit means that the classifier is already trained and we are just calibrating it
        # we use isotonic regression as it is more suitable for binary classification
        clf = CalibratedClassifierCV(estimator=best_base_clf, method='isotonic', cv="prefit")
        clf.fit(x_train_processed, new_y_train)

        # Identify pseudo-class index and known indices.
        class_labels = clf.classes_
        pseudo_class = max_known + 1
        known_indices = np.where(class_labels != pseudo_class)[0]

        def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_test_scaled = preprocessor.transform(x_test)
            probs = clf.predict_proba(x_test_scaled)
            known_probs = probs[:, known_indices]
            y_score = np.max(known_probs, axis=1)
            y_pred_all = clf.predict(x_test_scaled)
            # Map predictions from pseudo-class back to unknown (-1)
            y_pred = np.where(y_pred_all == pseudo_class, -1, y_pred_all)
            return y_pred, y_score

        return spl_predict_fn

    except Exception as e:
        logging.error("Error in SPL training: %s", e)
        raise

def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    try:
        # Preprocessing
        preprocessor = make_pipeline(RobustScaler(), Normalizer(norm='l2'))
        x_train_scaled = preprocessor.fit_transform(x_train)

        # Remap unknown samples using GMM (tune number of clusters separately if desired)
        known_mask = (y_train != -1)
        max_known = int(np.max(y_train[known_mask])) if np.any(known_mask) else 0
        new_y_train = np.copy(y_train)
        unknown_mask = (y_train == -1)
        
        # If there are unknown samples, cluster them using GMM and assign new labels
        if np.sum(unknown_mask) > 0:
            x_unknown = x_train_scaled[unknown_mask]
            best_gmm = None
            best_bic = np.inf # bio = Bayesian Information Criterion (lower is better)
            for n_components in [2, 3, 5, 7, 11]:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(x_unknown)
                bic = gmm.bic(x_unknown)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            cluster_labels = best_gmm.predict(x_unknown)
            new_unknown_labels = max_known + 1 + cluster_labels
            new_y_train[unknown_mask] = new_unknown_labels
            
        # if np.sum(unknown_mask) > 0:
        #     x_unknown = x_train_scaled[unknown_mask]
            
        #     # Try multiple eps values for DBSCAN
        #     best_n_clusters = 0
        #     best_labels = None
            
        #     for eps in [0.1, 0.2, 0.3, 0.5]:
        #         dbscan = DBSCAN(eps=eps, min_samples=max(2, int(0.05 * len(x_unknown))))
        #         labels = dbscan.fit_predict(x_unknown)
        #         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
        #         if n_clusters > best_n_clusters:
        #             best_n_clusters = n_clusters
        #             best_labels = labels
            
        #     # Remap cluster labels to new pseudo-classes
        #     if best_labels is not None and best_n_clusters > 0:
        #         cluster_labels = best_labels
        #         new_unknown_labels = max_known + 1 + cluster_labels
        #         new_unknown_labels[cluster_labels == -1] = -1  # Keep noise points as unknown
        #         new_y_train[unknown_mask] = new_unknown_labels

        # Tune a KNN classifier using GridSearchCV.
        knn = KNeighborsClassifier(weights='distance')
        param_grid = {
            'n_neighbors': [3, 5, 7, 11],
            'metric': ['cosine', 'euclidean', 'manhattan']
        }
        grid_search = GridSearchCV(
            knn, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=2
        )
        grid_search.fit(x_train_scaled, new_y_train)
        best_knn = grid_search.best_estimator_

        # Set adaptive thresholds for each class.
        thresholds = {}
        # confidence_thresholds = {}
        # confidence_threshold = 0.5
        
        # Define percentiles: for known classes, higher percentile (more tolerant),
        # and for unknown classes, lower percentile to trigger unknown classification more readily
        known_percentile = 95
        unknown_percentile = 86
        for label in np.unique(new_y_train):
            class_samples = x_train_scaled[new_y_train == label]
            if len(class_samples) > 0:
                distances, _ = best_knn.kneighbors(class_samples)
                if label <= max_known:
                    thresholds[label] = np.percentile(distances, known_percentile)
                    # confidence_thresholds[label] = confidence_threshold
                else:
                    thresholds[label] = np.percentile(distances, unknown_percentile)
                    # confidence_thresholds[label] = confidence_threshold

        # Global threshold for unknowns
        unknown_class_samples = x_train_scaled[new_y_train > max_known]
        if len(unknown_class_samples) > 0:
            global_unknown_threshold = np.percentile(
                best_knn.kneighbors(unknown_class_samples)[0], unknown_percentile
            )
        else:
            global_unknown_threshold = None

        def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_test_scaled = preprocessor.transform(x_test)
            distances, indices = best_knn.kneighbors(x_test_scaled)
            predictions = best_knn.predict(x_test_scaled)
            confidence_scores = np.zeros(len(x_test))
            
            for i, (distances, pred_label) in enumerate(zip(distances, predictions)):
                avg_distance = distances.mean()
                # get appropriate threshold for this class
                threshold = thresholds.get(
                    pred_label, 
                    global_unknown_threshold or np.inf
                )
                # convert distance to confidence score
                if threshold > 0:
                    confidence = 1-(avg_distance/threshold)
                else:
                    confidence = 1.0
                
                # applying distance threshold to detect unknowns
                # if avg_distance > threshold or confidence < confidence_threshold:
                if avg_distance > threshold:
                    predictions[i] = -1
                    confidence = max(0, confidence)  # ensure non-negative scores
                
                confidence_scores[i] = confidence
                
                
            return predictions, confidence_scores

        return mpl_predict_fn

    except Exception as e:
        logging.error("Error in MPL training: %s", e)
        raise


def load_challenge_validation_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge validation data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.chal_val_data, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y

def auc_roc(true_labels: np.ndarray, scores: np.ndarray) -> float:
    binary_labels = (true_labels != -1).astype(int)
    return roc_auc_score(binary_labels, scores)

def dir_far(true_labels: np.ndarray, scores: np.ndarray, far_target: float) -> float:
    known_mask = (true_labels != -1)
    unknown_scores = scores[~known_mask]
    if len(unknown_scores) == 0:
        return 0.0
    threshold = np.percentile(unknown_scores, 100 * (1 - far_target))
    return np.mean(scores[known_mask] >= threshold)

def rank1_accuracy(true_labels: np.ndarray, predictions: np.ndarray) -> float:
    known_mask = (true_labels != -1)
    if not np.any(known_mask):
        return 0.0
    return np.mean(true_labels[known_mask] == predictions[known_mask])

def main():
    try:
        # ========== Data Preparation ==========
        update_progress("Loading validation data", 10)
        features, labels = load_challenge_validation_data()

        update_progress("Splitting data into train/test sets", 20)
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # ========== Model Training ==========
        update_progress("Training SPL model", 40)
        spl_predictor = spl_training(x_train, y_train)

        update_progress("Training MPL model", 60)
        mpl_predictor = mpl_training(x_train, y_train)

        # ========== Model Evaluation ==========
        for model_idx, (name, predictor) in enumerate(zip(
            ["SPL", "MPL"], [spl_predictor, mpl_predictor]
        ), start=1):
            update_progress(f"Evaluating {name} model", 80)
            predictions, scores = predictor(x_test)
            
            # Calculate metrics
            accuracy = np.mean(y_test == predictions)
            auc = auc_roc(y_test, scores)
            dir_01 = dir_far(y_test, scores, 0.01)
            dir_10 = dir_far(y_test, scores, 0.10)
            rank1 = rank1_accuracy(y_test, predictions)

            # Log results
            logging.info(f"{name} Model Results:")
            logging.info(" - Basic Accuracy: %.4f", accuracy)
            logging.info(" - ROC AUC: %.4f", auc)
            logging.info(" - DIR@FAR=0.01: %.4f", dir_01)
            logging.info(" - DIR@FAR=0.10: %.4f", dir_10)
            logging.info(" - Rank-1 Accuracy: %.4f", rank1)
            logging.info("-" * 50)

        update_progress("Processing complete", 100)

    except Exception as e:
        logging.error("Main execution failed: %s", e)

        
if __name__ == "__main__":
    main()
