import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np

from cvproj_exc4.classifier import NearestNeighborClassifier
from cvproj_exc4.config import Config
from cvproj_exc4.evaluation import OpenSetEvaluation


def find_optimal_thresholds(results):
    # Unpack results
    false_alarm_rates = results["false_alarm_rates"]
    identification_rates = results["identification_rates"]
    similarity_thresholds = results["similarity_thresholds"]
    
    # Scenario 1: FAR ≤ 1% with maximum identification rate
    far_mask = false_alarm_rates <= 0.01
    far_indices = np.where(far_mask)[0]
    best_far_idx = far_indices[np.argmax(np.array(identification_rates)[far_mask])]
    
    scenario1 = {
        "similarity_threshold": similarity_thresholds[best_far_idx],
        "false_alarm_rate": false_alarm_rates[best_far_idx],
        "identification_rate": identification_rates[best_far_idx]
    }
    
    # Scenario 2: DIR ≥ 90% with minimum false alarms
    dir_mask = np.array(identification_rates) >= 0.9
    if np.any(dir_mask):
        dir_indices = np.where(dir_mask)[0]
        best_dir_idx = dir_indices[np.argmin(np.array(false_alarm_rates)[dir_mask])]
        
        scenario2 = {
            "similarity_threshold": similarity_thresholds[best_dir_idx],
            "false_alarm_rate": false_alarm_rates[best_dir_idx],
            "identification_rate": identification_rates[best_dir_idx]
        }
    else:
        scenario2 = "90% identification rate not achievable with current model"
    
    return scenario1, scenario2

def main():
    # The range of the false alarm rate in logarithmic space to draw DIR curves.
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

    # Pickle files containing embeddings and corresponding class labels for the
    # training and the test dataset.
    train_data_file = Config.eval_train_data
    test_data_file = Config.eval_test_data

    # We use a nearest neighbor classifier for this evaluation.
    classifier = NearestNeighborClassifier()

    # Prepare a new evaluation instance and feed training and test data into this evaluation.
    evaluation = OpenSetEvaluation(
        classifier=classifier, false_alarm_rate_range=false_alarm_rate_range
    )
    evaluation.prepare_input_data(train_data_file, test_data_file)

    # Run the evaluation and retrieve the performance measures (identification rates and
    # false alarm rates) on the test dataset.
    results = evaluation.run()

    # Find optimal thresholds
    scenario1, scenario2 = find_optimal_thresholds(results)

    # Print the results
    print("\nOptimal Thresholds Found:")
    print("\nScenario 1 (FAR ≤ 1%):")
    print(f"Similarity Threshold: {scenario1['similarity_threshold']:.4f}")
    print(f"False Alarm Rate: {scenario1['false_alarm_rate']:.4%}")
    print(f"Identification Rate: {scenario1['identification_rate']:.4%}")
    
    print("\nScenario 2 (DIR ≥ 90%):")
    if isinstance(scenario2, dict):
        print(f"Similarity Threshold: {scenario2['similarity_threshold']:.4f}")
        print(f"False Alarm Rate: {scenario2['false_alarm_rate']:.4%}")
        print(f"Identification Rate: {scenario2['identification_rate']:.4%}")
    else:
        print(scenario2)

    # Plot the DIR curve.
    plt.semilogx(
        false_alarm_rate_range,
        results["identification_rates"],
        markeredgewidth=1,
        linewidth=3,
        linestyle="--",
        color="blue",
    )
    plt.grid(True)
    plt.axis(
        [false_alarm_rate_range[0], false_alarm_rate_range[len(false_alarm_rate_range) - 1], 0, 1]
    )
    plt.xlabel("False alarm rate")
    plt.ylabel("Identification rate")
    plt.show()


if __name__ == "__main__":
    main()
