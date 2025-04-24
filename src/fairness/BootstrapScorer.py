import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from collections import defaultdict
from scipy import stats
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


class FairnessScorer:
    def __init__(self, n_iterations=100, random_seed=0):
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.unprivileged_groups = [
            {'Sex_protected': 0}, {'Ethnicity_protected': 0},
            {'Platform_protected': 0}, {'Preferred_hand_protected': 0}
        ]
        self.privileged_groups = [
            {'Sex_protected': 1}, {'Ethnicity_protected': 1},
            {'Platform_protected': 1}, {'Preferred_hand_protected': 1}
        ]

    def _calculate_metrics_from_cm(self, cm):
        TP, FP, TN, FN = cm['TP'], cm['FP'], cm['TN'], cm['FN']
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1, recall, specificity, precision

    def _calculate_eq_odds_diff(self, unpriv_cm, priv_cm):
        def tpr(cm): return cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) != 0 else 0
        def fpr(cm): return cm['FP'] / (cm['FP'] + cm['TN']) if (cm['FP'] + cm['TN']) != 0 else 0
        return max(abs(tpr(unpriv_cm) - tpr(priv_cm)), abs(fpr(unpriv_cm) - fpr(priv_cm)))

    def _calculate_auc(self, df, group_col):
        priv = df[df[group_col] == 1]
        unpriv = df[df[group_col] == 0]
        auc_priv = roc_auc_score(priv['PD_status'], priv['prediction']) if len(priv['PD_status'].unique()) == 2 else np.nan
        auc_unpriv = roc_auc_score(unpriv['PD_status'], unpriv['prediction']) if len(unpriv['PD_status'].unique()) == 2 else np.nan
        return auc_priv, auc_unpriv

    def _calculate_p_value(self, unpriv, priv):
        return stats.ttest_ind(unpriv, priv).pvalue

    def score(self, clf, X, y):
        f1_scores, sens_scores, spec_scores = [], [], []
        metrics = defaultdict(list)
        half_size = len(X) // 2

        for i in range(self.n_iterations):
            X_res, y_res = resample(X, y, replace=True, random_state=i + self.random_seed)
            y_pred = clf.predict(X_res)
            f1_scores.append(f1_score(y_res, y_pred))
            sens_scores.append(recall_score(y_res, y_pred, pos_label=1))
            spec_scores.append(recall_score(y_res, y_pred, pos_label=0))

            for unpriv, priv in zip(self.unprivileged_groups, self.privileged_groups):
                group = list(unpriv.keys())[0]
                data = X.copy()
                data['PD_status'] = y
                unpriv_data = resample(data[data[group] == 0], replace=True, n_samples=half_size, random_state=i)
                priv_data = resample(data[data[group] == 1], replace=True, n_samples=half_size, random_state=i)
                combined = pd.concat([unpriv_data, priv_data])
                y_combined = combined['PD_status']
                X_combined = combined.drop(columns='PD_status')
                y_pred_comb = clf.predict(X_combined)
                y_proba_comb = clf.predict_proba(X_combined)[:, 1]

                X_combined['PD_status'] = y_combined
                X_combined['prediction'] = y_proba_comb

                dataset = BinaryLabelDataset(df=X_combined, label_names=["PD_status"],
                                             favorable_label=1, unfavorable_label=0,
                                             protected_attribute_names=list(unpriv.keys()))
                pred = dataset.copy()
                pred.labels = y_pred_comb.reshape(-1, 1)

                metric = ClassificationMetric(dataset, pred, unprivileged_groups=[unpriv], privileged_groups=[priv])
                unpriv_cm = metric.binary_confusion_matrix(privileged=False)
                priv_cm = metric.binary_confusion_matrix(privileged=True)

                u_f1, u_sens, u_spec, u_prec = self._calculate_metrics_from_cm(unpriv_cm)
                p_f1, p_sens, p_spec, p_prec = self._calculate_metrics_from_cm(priv_cm)
                di = metric.disparate_impact()
                eq_opp = u_sens / p_sens if p_sens != 0 else 0
                eq_odds = self._calculate_eq_odds_diff(unpriv_cm, priv_cm)
                auc_p, auc_u = self._calculate_auc(X_combined, group)

                metrics[f"unprivileged_{group}_f1"].append(u_f1)
                metrics[f"unprivileged_{group}_sensitivity"].append(u_sens)
                metrics[f"unprivileged_{group}_specificity"].append(u_spec)
                metrics[f"unprivileged_{group}_precision"].append(u_prec)
                metrics[f"privileged_{group}_f1"].append(p_f1)
                metrics[f"privileged_{group}_sensitivity"].append(p_sens)
                metrics[f"privileged_{group}_specificity"].append(p_spec)
                metrics[f"privileged_{group}_precision"].append(p_prec)
                metrics[f"{group}_disparate_impact"].append(di)
                metrics[f"{group}_equal_opportunity"].append(eq_opp)
                metrics[f"{group}_equalized_odds_difference"].append(eq_odds)
                metrics[f"unprivileged_{group}_auc"].append(auc_u)
                metrics[f"privileged_{group}_auc"].append(auc_p)

        result = {
            'f1_score_mean': np.mean(f1_scores), 'f1_score_std': np.std(f1_scores),
            'sensitivity_mean': np.mean(sens_scores), 'sensitivity_std': np.std(sens_scores),
            'specificity_mean': np.mean(spec_scores), 'specificity_std': np.std(spec_scores)
        }

        for unpriv, priv in zip(self.unprivileged_groups, self.privileged_groups):
            key = list(unpriv.keys())[0]
            result[f"{key}_f1_p"] = self._calculate_p_value(metrics[f"unprivileged_{key}_f1"],
                                                            metrics[f"privileged_{key}_f1"])
            for metric in ['f1', 'auc', 'sensitivity', 'specificity', 'precision']:
                for group in ['unprivileged', 'privileged']:
                    k = f"{group}_{key}_{metric}"
                    result[f"{k}_mean"] = np.mean(metrics[k])
                    result[f"{k}_std"] = np.std(metrics[k])
            for metric in ['disparate_impact', 'equal_opportunity', 'equalized_odds_difference']:
                k = f"{key}_{metric}"
                result[f"{k}_mean"] = np.mean(metrics[k])
                result[f"{k}_std"] = np.std(metrics[k])

        return result


def bootstrap_scorer(clf, X, y):
    return FairnessScorer().score(clf, X, y)