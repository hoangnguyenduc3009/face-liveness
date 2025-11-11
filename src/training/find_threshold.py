import numpy as np
from sklearn.metrics import roc_curve

def find_best_threshold(y_true, y_pred_probs):
    """
    Find the best threshold for classification to maximize Youden's J statistic.
    """
    if len(np.unique(y_true)) < 2:
        # Handle case where there is only one class in y_true
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_thresh_idx]
    
    # In cases where roc_curve returns inf, we should default to 0.5
    if best_thresh == np.inf:
        return 0.5, 0.0
        
    return best_thresh, j_scores[best_thresh_idx]
