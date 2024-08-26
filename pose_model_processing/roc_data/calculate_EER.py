import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


def load_roc_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    true_labels = df['TrueLabels'].values
    pred_probs = df['PredProbs'].values
    return true_labels, pred_probs


def calculate_eer(true_labels, pred_probs):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer, eer_threshold


def calculate_eer_for_models(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith("_roc_data.csv"):
            model_name = filename.replace('_roc_data.csv', '')
            file_path = os.path.join(directory, filename)

            # Load the data from CSV
            true_labels, pred_probs = load_roc_data_from_csv(file_path)

            # Calculate the EER
            eer, eer_threshold = calculate_eer(true_labels, pred_probs)
            print(f"Model: {model_name}")
            print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}\n")


# Example usage:
calculate_eer_for_models(directory=".")
