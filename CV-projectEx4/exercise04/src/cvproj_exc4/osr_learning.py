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
        # Preprocessing: Use RobustScaler and Normalizer
        preprocessor = make_pipeline(RobustScaler(), Normalizer(norm='l2'))
        x_train_scaled = preprocessor.fit_transform(x_train)

        # Remap unknown samples using GMM (tune number of clusters separately if desired)
        known_mask = (y_train != -1)
        max_known = int(np.max(y_train[known_mask])) if np.any(known_mask) else 0
        new_y_train = np.copy(y_train)
        unknown_mask = (y_train == -1)
        if np.sum(unknown_mask) > 0:
            x_unknown = x_train_scaled[unknown_mask]
            best_gmm = None
            best_bic = np.inf
            for n_components in [2, 3, 5, 7, 10]:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(x_unknown)
                bic = gmm.bic(x_unknown)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            cluster_labels = best_gmm.predict(x_unknown)
            new_unknown_labels = max_known + 1 + cluster_labels
            new_y_train[unknown_mask] = new_unknown_labels

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
        # Define percentiles: for known classes, use a higher percentile (more tolerant),
        # and for unknown classes, use a lower percentile to trigger unknown classification more readily.
        known_percentile = 95
        unknown_percentile = 86
        for label in np.unique(new_y_train):
            class_samples = x_train_scaled[new_y_train == label]
            if len(class_samples) > 0:
                distances, _ = best_knn.kneighbors(class_samples)
                if label <= max_known:
                    thresholds[label] = np.percentile(distances, known_percentile)
                else:
                    thresholds[label] = np.percentile(distances, unknown_percentile)

        # Global threshold for unknowns (optional): compute using all samples with labels > max_known.
        unknown_class_samples = x_train_scaled[new_y_train > max_known]
        if len(unknown_class_samples) > 0:
            global_unknown_threshold = np.percentile(
                best_knn.kneighbors(unknown_class_samples)[0], unknown_percentile
            )
        else:
            global_unknown_threshold = None

        def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_test_scaled = preprocessor.transform(x_test)
            distances, _ = best_knn.kneighbors(x_test_scaled)
            y_pred = best_knn.predict(x_test_scaled)
            y_score = np.zeros(len(x_test))
            for i, (dist, pred_label) in enumerate(zip(distances, y_pred)):
                mean_dist = dist.mean()
                # Get threshold: if label not found, use global_unknown_threshold.
                threshold = thresholds.get(pred_label, global_unknown_threshold)
                if threshold is not None and mean_dist > threshold:
                    y_pred[i] = -1
                    y_score[i] = 1 - (mean_dist / threshold)
                else:
                    y_score[i] = 1 - (mean_dist / threshold if threshold and threshold > 0 else 1)
            return y_pred, y_score

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
    print('Running main')
    try:
        update_progress("Loading validation data", 10)
        x, y = load_challenge_validation_data()

        update_progress("Splitting data into training and test sets", 20)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        update_progress("Training SPL model", 40)
        spl_predict_fn = spl_training(x_train, y_train)

        update_progress("Training MPL model", 60)
        mpl_predict_fn = mpl_training(x_train, y_train)

        for idx, predict_fn in enumerate((spl_predict_fn, mpl_predict_fn), start=1):
            update_progress(f"Evaluating model {idx}", 80)
            y_pred, y_score = predict_fn(x_test)
            dummy_acc = np.equal(y_test, y_pred).sum() / len(x_test)
            logging.info("Dummy Accuracy: %.4f", dummy_acc)
            logging.info("AUC-ROC: %.4f", auc_roc(y_test, y_score))
            logging.info("DIR@FAR=0.01: %.4f", dir_far(y_test, y_score, 0.01))
            logging.info("DIR@FAR=0.10: %.4f", dir_far(y_test, y_score, 0.10))
            logging.info("Rank-1 Accuracy: %.4f", rank1_accuracy(y_test, y_pred))
            logging.info("--------------------------------------------------")

        update_progress("Processing complete", 100)

    except Exception as e:
        logging.error("An error occurred in main: %s", e)

        
if __name__ == "__main__":
    main()
