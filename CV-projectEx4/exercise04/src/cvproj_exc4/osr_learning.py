from collections.abc import Callable

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cvproj_exc4.config import Config 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'spl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    import sklearn.pipeline as skpipe
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    estimator_pipeline = skpipe.Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )

    estimator_pipeline.fit(x_train, y_train)


    
    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.
        y_pred = None
        y_score = None

        y_pred = estimator_pipeline.predict(x_test)
        y_score = estimator_pipeline.predict_proba(x_test)[:, 1]

        return y_pred, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'mpl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    import sklearn.pipeline as skpipe
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    estimator_pipeline = skpipe.Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )

    estimator_pipeline.fit(x_train, y_train)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.
        y_pred = None
        y_score = None

        y_pred = estimator_pipeline.predict(x_test)
        y_score = estimator_pipeline.predict_proba(x_test)[:, 1]

        return y_pred, y_score

    return mpl_predict_fn


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


def auc_roc(y_test, y_score):
    y_test = (y_test != -1).astype(int)
    return roc_auc_score(y_test, y_score)


def dir_far(y_test, y_score, far_target):
    kc_indices = (y_test != -1)
    kuc_indices = (y_test == -1)

    unknown_scores = y_score[kuc_indices]
    threshold = np.percentile(unknown_scores, 100 * (1 - far_target))

    known_scores = y_score[kc_indices]
    true_labels = y_test[kc_indices]

    predicted_positive = (known_scores >= threshold)
    dir = np.sum(predicted_positive) / len(true_labels)
    return dir


def rank1_accuracy(y_test, y_pred):
    kc_indices = (y_test != -1)
    y_test_known = y_test[kc_indices]
    y_pred_known = y_pred[kc_indices]

    #rank-1 accuracy for KC
    correct_predictions = (y_test_known == y_pred_known).sum()
    if len(y_test_known) > 0:
        rank1_accuracy = correct_predictions / len(y_test_known)
    else:
        rank1_accuracy = 0.0

    return rank1_accuracy


def main():
    x_train, y_train = load_challenge_validation_data()

    
    x_train, y_train = load_challenge_validation_data()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # TODO: implement
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So
    #       please make sure that this call (besides the unit tests) does what it is supposed to do.
    # x_test = np.random.rand(50, x_train.shape[1])
    # y_test = np.random.randint(-1, 5, 50)
    # for predict_fn in (spl_predict_fn, mpl_predict_fn):
    #     y_pred, y_score = predict_fn(x_test)
    #     print("Dummy acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))

    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Dummy acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))
        print("AUC-ROC: {}".format(auc_roc(y_test, y_score)))
        print("DIR@FAR=0.01: {}".format(dir_far(y_test, y_score, 0.01)))
        print("DIR@FAR=0.10: {}".format(dir_far(y_test, y_score, 0.10)))
        print("Rank-1 Accuracy: {}".format(rank1_accuracy(y_test, y_pred)))


if __name__ == "__main__":
    main()
