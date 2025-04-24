import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
from sklearn.utils import resample


class BootstrapScorer:
    def __init__(self, n_iterations=1000, random_seed=0):
        self.n_iterations = n_iterations
        self.random_seed = random_seed

    def score(self, clf, X, y):
        f1_scores = []
        sensitivity_scores = []
        specificity_scores = []
        accuracy_scores = []
        auc_scores = []

        for i in range(self.n_iterations):
            X_resampled, y_resampled = resample(X, y, replace=True, random_state=i + self.random_seed)
            y_pred = clf.predict(X_resampled)
            y_pred_proba = clf.predict_proba(X_resampled)[:, 1]

            f1_scores.append(f1_score(y_resampled, y_pred))
            sensitivity_scores.append(recall_score(y_resampled, y_pred, pos_label=1))
            specificity_scores.append(recall_score(y_resampled, y_pred, pos_label=0))
            accuracy_scores.append(accuracy_score(y_resampled, y_pred))
            auc_scores.append(roc_auc_score(y_resampled, y_pred_proba))

        return {
            'f1_score_mean': np.mean(f1_scores), 'f1_score_std': np.std(f1_scores),
            'accuracy_mean': np.mean(accuracy_scores), 'accuracy_std': np.std(accuracy_scores),
            'auc_mean': np.mean(auc_scores), 'auc_std': np.std(auc_scores),
            'sensitivity_mean': np.mean(sensitivity_scores), 'sensitivity_std': np.std(sensitivity_scores),
            'specificity_mean': np.mean(specificity_scores), 'specificity_std': np.std(specificity_scores)
        }


def bootstrap_scorer(clf, X, y):
    return BootstrapScorer().score(clf, X, y)
