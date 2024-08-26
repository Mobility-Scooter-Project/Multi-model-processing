import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_roc_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    true_labels = df['TrueLabels'].values
    pred_probs = df['PredProbs'].values
    return true_labels, pred_probs

def plot_multiple_roc_curves(directory="."):
    plt.figure()
    for filename in os.listdir(directory):
        if filename.endswith("_roc_data.csv"):
            model_name = filename.replace('_roc_data.csv', '')
            file_path = os.path.join(directory, filename)
            true_labels, pred_probs = load_roc_data_from_csv(file_path)
            fpr, tpr, _ = roc_curve(true_labels, pred_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    directory = "."
    plot_multiple_roc_curves(directory=directory)
