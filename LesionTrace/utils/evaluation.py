from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, cohen_kappa_score
from pathlib import Path
import numpy as np
import pandas as pd


# Function to calculate Specificity metric
def specificity(class_ground_truth, class_prediction):
    """
    Calculates the Specificity metric for a binary classification problem.

    Args:
        class_ground_truth (array-like): Array of true class labels.
        class_prediction (array-like): Array of predicted class labels.

    Returns:
        float: The average Specificity score.
    """

    eps = 0.000001  # Add a small value to avoid division by zero
    cnf_matrix = confusion_matrix(class_ground_truth, class_prediction)

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Cast all values to float to avoid type errors
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Calculate Specificity for each class and average them
    spe = TN / (TN + FP + eps)
    spe = spe.mean()

    return spe


# Function to calculate multiple evaluation metrics
def score_aggregates(labels, predictions):
    """
    Calculates F1 score, Matthews Correlation Coefficient, and Specificity for a classification task.

    Args:
        df_solution (pandas.DataFrame): DataFrame containing ground truth labels.
        df_prediction (pandas.DataFrame): DataFrame containing predicted labels.

    Returns:
        dict: Dictionary containing F1 score, Matthews Correlation Coefficient, and Specificity metrics.
    """

    # Calculate evaluation metrics
    return {
        "F1_score": f1_score(labels, predictions, average="micro"),
        "Rk-correlation": matthews_corrcoef(labels, predictions),
        "Specificity": specificity(labels, predictions),
    }
