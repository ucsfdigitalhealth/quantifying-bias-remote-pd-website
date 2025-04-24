import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_validate
from joblib import dump
from src.fairness.BootstrapScorer import bootstrap_scorer
from src.fairness.DataPreprocessor import get_processed_data
from config import PROJECT_DIR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from yaspin import yaspin
from yaspin.spinners import Spinners


warnings.filterwarnings("ignore")

class FairnessEvaluator:
    def __init__(self, X, y, prefix, out_dir="analysis"):
        self.models = self._initialize_models()
        self.X = X
        self.y = y
        self.prefix = prefix
        self.results = {}
        self.out_dir = os.path.join(PROJECT_DIR, out_dir, self.prefix)
        os.makedirs(self.out_dir, exist_ok=True)

    def _initialize_models(self):
        return {
            "Logistic Regression": LogisticRegression(solver='liblinear', C=10, penalty='l2'),
            "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='log2',
                                                    min_samples_leaf=1, min_samples_split=6, class_weight='balanced'),
            "LinearSVC": CalibratedClassifierCV(LinearSVC(C=1, penalty='l1', dual=False), method='sigmoid'),
            "Random Forest": RandomForestClassifier(max_depth=5, max_features='log2', min_samples_leaf=1,
                                                    min_samples_split=5, n_estimators=100, class_weight='balanced'),
            "XGB Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(verbosity=-1)
        }
    
    def view_result(self, result, title):
        print(result)
        bold = "\033[1m"
        reset = "\033[0m"
        print(f"{bold}{title} (Mean of 5-fold Cross Validation with 1000 Bootstrap Sampling at Each Fold){reset}")
        print(f"{bold}Model Performance{reset}")
        print(f"F1 Score: {result['test_f1_score_mean'].mean():.2f} ({np.std(result['test_f1_score_mean']):.2f})")
        print(f"Sensitivity: {result['test_sensitivity_mean'].mean():.2f} ({np.std(result['test_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_specificity_mean'].mean():.2f} ({np.std(result['test_specificity_mean']):.2f})")


        print(f"{bold}Fairness Metrics{reset}")

        p_value = np.mean(result['test_Sex_protected_f1_p'])
        p_decision = "Significant" if p_value < 0.05 else "Not Significant"
        print(f"{bold}Gender{reset}")
        print(f"Disparate Impact: {np.mean(result['test_Sex_protected_disparate_impact_mean']):.2f} ({np.std(result['test_Sex_protected_disparate_impact_mean']):.2f})")
        print(f"Equal Opportunity: {np.mean(result['test_Sex_protected_equal_opportunity_mean']):.2f} ({np.std(result['test_Sex_protected_equal_opportunity_mean']):.2f})")
        print(f"Equalied Odds Difference: {np.mean(result['test_Sex_protected_equalized_odds_difference_mean']):.2f} ({np.std(result['test_Sex_protected_equalized_odds_difference_mean']):.2f})")
        print(f"Significant Difference: {p_decision}")


        print(f"{bold}Male{reset}")
        print(f"F1 Score: {result['test_privileged_Sex_protected_f1_mean'].mean():.2f} ({np.std(result['test_privileged_Sex_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_privileged_Sex_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_privileged_Sex_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_privileged_Sex_protected_specificity_mean'].mean():.2f} ({np.std(result['test_privileged_Sex_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_privileged_Sex_protected_precision_mean'].mean():.2f} ({np.std(result['test_privileged_Sex_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_privileged_Sex_protected_auc_mean']):.2f} ({np.nanstd(result['test_privileged_Sex_protected_auc_mean']):.2f})")
        print(f"{bold}Female{reset}")
        print(f"F1 Score: {result['test_unprivileged_Sex_protected_f1_mean'].mean():.2f} ({np.std(result['test_unprivileged_Sex_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_unprivileged_Sex_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Sex_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_unprivileged_Sex_protected_specificity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Sex_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_unprivileged_Sex_protected_precision_mean'].mean():.2f} ({np.std(result['test_unprivileged_Sex_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_unprivileged_Sex_protected_auc_mean']):.2f} ({np.nanstd(result['test_unprivileged_Sex_protected_auc_mean']):.2f})")

        print("")

        p_value = np.mean(result['test_Ethnicity_protected_f1_p'])
        p_decision = "Significant" if p_value < 0.05 else "Not Significant"

        print(f"{bold}Race{reset}")
        print(f"Disparate Impact: {np.mean(result['test_Ethnicity_protected_disparate_impact_mean']):.2f} ({np.std(result['test_Ethnicity_protected_disparate_impact_mean']):.2f})")
        print(f"Equal Opportunity: {np.mean(result['test_Ethnicity_protected_equal_opportunity_mean']):.2f} ({np.std(result['test_Ethnicity_protected_equal_opportunity_mean']):.2f})")
        print(f"Equalied Odds Difference: {np.mean(result['test_Ethnicity_protected_equalized_odds_difference_mean']):.2f} ({np.std(result['test_Ethnicity_protected_equalized_odds_difference_mean']):.2f})")

        print(f"Significant Difference: {p_decision}")

        print(f"{bold}White{reset}")
        print(f"F1 Score: {result['test_privileged_Ethnicity_protected_f1_mean'].mean():.2f} ({np.std(result['test_privileged_Ethnicity_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_privileged_Ethnicity_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_privileged_Ethnicity_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_privileged_Ethnicity_protected_specificity_mean'].mean():.2f} ({np.std(result['test_privileged_Ethnicity_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_privileged_Ethnicity_protected_precision_mean'].mean():.2f} ({np.std(result['test_privileged_Ethnicity_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_privileged_Ethnicity_protected_auc_mean']):.2f} ({np.nanstd(result['test_privileged_Ethnicity_protected_auc_mean']):.2f})")
        print(f"{bold}Non-White{reset}")
        print(f"F1 Score: {result['test_unprivileged_Ethnicity_protected_f1_mean'].mean():.2f} ({np.std(result['test_unprivileged_Ethnicity_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_unprivileged_Ethnicity_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Ethnicity_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_unprivileged_Ethnicity_protected_specificity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Ethnicity_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_unprivileged_Ethnicity_protected_precision_mean'].mean():.2f} ({np.std(result['test_unprivileged_Ethnicity_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_unprivileged_Ethnicity_protected_auc_mean']):.2f} ({np.nanstd(result['test_unprivileged_Ethnicity_protected_auc_mean']):.2f})")
        

        print("")
        
        p_value = np.mean(result['test_Platform_protected_f1_p'])
        p_decision = "Significant" if p_value < 0.05 else "Not Significant"

        print(f"{bold}Device Type{reset}")
        print(f"Disparate Impact: {np.mean(result['test_Platform_protected_disparate_impact_mean']):.2f} ({np.std(result['test_Platform_protected_disparate_impact_mean']):.2f})")
        print(f"Equal Opportunity: {np.mean(result['test_Platform_protected_equal_opportunity_mean']):.2f} ({np.std(result['test_Platform_protected_equal_opportunity_mean']):.2f})")
        print(f"Equalied Odds Difference: {np.mean(result['test_Platform_protected_equalized_odds_difference_mean']):.2f} ({np.std(result['test_Platform_protected_equalized_odds_difference_mean']):.2f})")
        print(f"Significant Difference: {p_decision}")

        print(f"{bold}Mac{reset}")
        print(f"F1 Score: {result['test_privileged_Platform_protected_f1_mean'].mean():.2f} ({np.std(result['test_privileged_Platform_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_privileged_Platform_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_privileged_Platform_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_privileged_Platform_protected_specificity_mean'].mean():.2f} ({np.std(result['test_privileged_Platform_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_privileged_Platform_protected_precision_mean'].mean():.2f} ({np.std(result['test_privileged_Platform_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_privileged_Platform_protected_auc_mean']):.2f} ({np.nanstd(result['test_privileged_Platform_protected_auc_mean']):.2f})")
        print(f"{bold}Windows{reset}")
        print(f"F1 Score: {result['test_unprivileged_Platform_protected_f1_mean'].mean():.2f} ({np.std(result['test_unprivileged_Platform_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_unprivileged_Platform_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Platform_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_unprivileged_Platform_protected_specificity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Platform_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_unprivileged_Platform_protected_precision_mean'].mean():.2f} ({np.std(result['test_unprivileged_Platform_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_unprivileged_Platform_protected_auc_mean']):.2f} ({np.nanstd(result['test_unprivileged_Platform_protected_auc_mean']):.2f})")


        print("")
        
        p_value = np.mean(result['test_Preferred_hand_protected_f1_p'])
        p_decision = "Significant" if p_value < 0.05 else "Not Significant"


        print(f"{bold}Dominant Hand{reset}")
        print(f"Disparate Impact: {np.mean(result['test_Preferred_hand_protected_disparate_impact_mean']):.2f} ({np.std(result['test_Preferred_hand_protected_disparate_impact_mean']):.2f})")
        print(f"Equal Opportunity: {np.mean(result['test_Preferred_hand_protected_equal_opportunity_mean']):.2f} ({np.std(result['test_Preferred_hand_protected_equal_opportunity_mean']):.2f})")
        print(f"Equalied Odds Difference: {np.mean(result['test_Preferred_hand_protected_equalized_odds_difference_mean']):.2f} ({np.std(result['test_Preferred_hand_protected_equalized_odds_difference_mean']):.2f})")
        print(f"Significant Difference: {p_decision}")

        print(f"{bold}Dominant Hand{reset}")
        print(f"F1 Score: {result['test_privileged_Preferred_hand_protected_f1_mean'].mean():.2f} ({np.std(result['test_privileged_Preferred_hand_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_privileged_Preferred_hand_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_privileged_Preferred_hand_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_privileged_Preferred_hand_protected_specificity_mean'].mean():.2f} ({np.std(result['test_privileged_Preferred_hand_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_privileged_Preferred_hand_protected_precision_mean'].mean():.2f} ({np.std(result['test_privileged_Preferred_hand_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_privileged_Preferred_hand_protected_auc_mean']):.2f} ({np.nanstd(result['test_privileged_Preferred_hand_protected_auc_mean']):.2f})")
        print(f"{bold}Left{reset}")
        print(f"F1 Score: {result['test_unprivileged_Preferred_hand_protected_f1_mean'].mean():.2f} ({np.std(result['test_unprivileged_Preferred_hand_protected_f1_mean']):.2f})")
        print(f"Sensitivity: {result['test_unprivileged_Preferred_hand_protected_sensitivity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Preferred_hand_protected_sensitivity_mean']):.2f})")
        print(f"Specificity: {result['test_unprivileged_Preferred_hand_protected_specificity_mean'].mean():.2f} ({np.std(result['test_unprivileged_Preferred_hand_protected_specificity_mean']):.2f})")
        print(f"Precision: {result['test_unprivileged_Preferred_hand_protected_precision_mean'].mean():.2f} ({np.std(result['test_unprivileged_Preferred_hand_protected_precision_mean']):.2f})")
        print(f"AUC: {np.nanmean(result['test_unprivileged_Preferred_hand_protected_auc_mean']):.2f} ({np.nanstd(result['test_unprivileged_Preferred_hand_protected_auc_mean']):.2f})")
       

    def evaluate_models(self):
        for name, model in self.models.items():
            with yaspin(text=f"Evaluating {name}...", color="cyan") as spinner:
                try:
                    result = cross_validate(
                        model,
                        self.X,
                        self.y,
                        cv=5,
                        scoring=bootstrap_scorer,
                        return_estimator=True,
                        return_train_score=True
                    )
                    self.results[name] = result
                    spinner.ok("✅")
                    self.view_result(result, name)
                except Exception as e:
                    spinner.fail("❌")
                    print(f"❌ Failed to train {name}: {e}")

    def plot_groupwise_bias(self, protected_attr='Sex_protected', metric='f1'):
        """Plots group-wise performance for protected attribute"""
        attr_map = {
            'Sex_protected': ('Male', 'Female'),
            'Ethnicity_protected': ('White', 'Non-White'),
            'Platform_protected': ('Mac', 'Windows'),
            'Preferred_hand_protected': ('Right', 'Left')
        }

        title_map = {
            'Sex_protected': 'Sex',
            'Ethnicity_protected': 'Race',
            'Platform_protected': 'Device',
            'Preferred_hand_protected': 'Dominant Hand'
        }

        if protected_attr not in attr_map:
            print(f"⚠️ Unsupported attribute: {protected_attr}")
            return

        labels = attr_map[protected_attr]
        privileged_col = f"test_privileged_{protected_attr}_{metric}_mean"
        unprivileged_col = f"test_unprivileged_{protected_attr}_{metric}_mean"

        group_bar = pd.DataFrame({
            'Algorithms': sum([[name, name] for name in self.results.keys()], []),
            'Group': [labels[0]] * len(self.results) + [labels[1]] * len(self.results),
            'Value': [np.mean(self.results[name][privileged_col]) for name in self.results] +
                     [np.mean(self.results[name][unprivileged_col]) for name in self.results],
            'Std': [np.std(self.results[name][privileged_col]) for name in self.results] +
                   [np.std(self.results[name][unprivileged_col]) for name in self.results]
        })

        fig, ax = plt.subplots(figsize=(15, 6))
        bar_width = 0.35
        indices = np.arange(len(self.results))
        r1 = indices
        r2 = [x + bar_width for x in r1]

        group1 = group_bar[group_bar['Group'] == labels[0]]
        group2 = group_bar[group_bar['Group'] == labels[1]]

        bars1 = ax.bar(r1, group1['Value'], width=bar_width, label=labels[0],
                       yerr=group1['Std'], capsize=5)
        bars2 = ax.bar(r2, group2['Value'], width=bar_width, label=labels[1],
                       yerr=group2['Std'], capsize=5)

        ax.set_xlabel('Algorithms', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax.set_title(f'Differences in {metric.upper()} Score by {title_map[protected_attr]}')
        ax.set_xticks([r + bar_width / 2 for r in range(len(self.results))])
        ax.set_xticklabels(group1['Algorithms'])

        def add_values(bars, data):
            for bar, (_, row) in zip(bars, data.iterrows()):
                yval = bar.get_height()
                err = row['Std']
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        yval + err + 0.02,
                        f"{round(yval, 2)}", ha='center', va='bottom', fontsize=10)

        add_values(bars1, group1)
        add_values(bars2, group2)
        max_y = max((group1['Value'] + group1['Std']).max(),
            (group2['Value'] + group2['Std']).max())
        ax.set_ylim(0, max_y + 0.1)
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(self.out_dir, f"{protected_attr}_{metric}_group_plot.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📊 Saved group-level bias plot to {save_path}")


def evaluate_fairness(race_upsample = False):
    print("🚀 Loading preprocessed data...")
    X, y = get_processed_data(race_upsample)

    prefix = "after_race" if race_upsample else "before_race"

    evaluator = FairnessEvaluator(X, y, prefix=prefix)
    evaluator.evaluate_models()
    protected_groups = ["Sex_protected", "Ethnicity_protected", "Platform_protected", "Preferred_hand_protected"]

    for attr in protected_groups:
        evaluator.plot_groupwise_bias(protected_attr=attr)
