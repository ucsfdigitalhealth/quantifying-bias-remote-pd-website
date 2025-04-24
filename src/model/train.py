import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from yaspin import yaspin
from yaspin.spinners import Spinners
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model.DataPreprocessor import get_processed_data
from config import PROJECT_DIR
from src.model.BootstrapScorer import bootstrap_scorer

from concurrent.futures import ProcessPoolExecutor, as_completed


warnings.filterwarnings("ignore")

class ParkinsonModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = self.initialize_models()
        self.results = {}

        os.makedirs(os.path.join(PROJECT_DIR, "weights"), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_DIR, "analysis"), exist_ok=True)

    def initialize_models(self):
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

    def train_all(self):
        print("🧠 Training models...\n")

        for model_name, model in self.models.items():
            with yaspin(text=f"Training {model_name}...", color="cyan") as spinner:
                try:
                    result = cross_validate(
                        model,
                        self.X_train,
                        self.y_train,
                        cv=5,
                        return_estimator=True,
                        return_train_score=True,
                        scoring=bootstrap_scorer
                    )

                    self.results[model_name] = result
                    best_model = max(
                        zip(result['train_f1_score_mean'], result['estimator']),
                        key=lambda x: x[0]
                    )[1]
                    model_filename = f"{model_name.replace(' ', '_').lower()}.joblib"
                    save_path = os.path.join(PROJECT_DIR, "weights", model_filename)
                    joblib.dump(best_model, save_path)

                    spinner.ok("✅")
                    print(f"\n✅ {model_name} Training Complete:")
                    print(f"  F1 Score:       {np.mean(result['test_f1_score_mean']):.4f} ± {np.std(result['test_f1_score_mean']):.4f}")
                    print(f"  AUROC:          {np.mean(result['test_auc_mean']):.4f} ± {np.std(result['test_auc_mean']):.4f}")
                    print(f"  Sensitivity:    {np.mean(result['test_sensitivity_mean']):.4f} ± {np.std(result['test_sensitivity_mean']):.4f}")
                    print(f"  Specificity:    {np.mean(result['test_specificity_mean']):.4f} ± {np.std(result['test_specificity_mean']):.4f}")

                except Exception as e:
                    spinner.fail("❌")
                    print(f"❌ Failed to train {model_name}: {e}")


    def summarize_and_plot(self):
        print("\n📊 Summarizing and plotting performance...")
        metrics = ["f1_score_mean", "auc_mean"]
        all_data = []

        for name, result in self.results.items():
            for metric in metrics:
                all_data.append({
                    "Algorithms": name,
                    "Metric": "F1 Score" if "f1" in metric else "AUROC",
                    "Mean": np.mean(result[f"test_{metric}"]),
                    "Std": np.std(result[f"test_{metric}"])
                })

        df = pd.DataFrame(all_data)

        sns.set_context("notebook", font_scale=1.2)
        sns.set_style("whitegrid")
        plt.figure(figsize=(18, 6))
        ax = sns.barplot(data=df, x="Mean", y="Algorithms", hue="Metric", palette=["#00a8ff", "#fbc531"])


        num_groups = len(ax.containers)


        for index, container in enumerate(ax.containers):
            container_indices = list(range(index, len(df), num_groups))
            for bar, row in zip(container, container_indices):
                data = df.iloc[row]
                width = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                label = f"{data['Mean']:.4f} ± {data['Std']:.4f}"
                ax.annotate(label, (width, y), xytext=(6, 0),
                            textcoords='offset points', ha='left', va='center', fontsize=8)



        max_score = df['Mean'].max() + df['Std'].max()
        plt.xlim(0, max_score + 0.05)

        plt.xlabel("Score")
        plt.ylabel("Algorithms")
        plt.title("Comparison of Models: F1 Score vs AUROC")
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIR, "analysis/training.png"))
        plt.show()
        print("📁 Saved plot to analysis/training.png")


def train_model():
    print("🚀 Loading preprocessed data...")
    X_train, X_test, y_train, y_test = get_processed_data()

    trainer = ParkinsonModelTrainer(X_train, y_train)

    print("🚀 Model Training Started...")
    trainer.train_all()
    trainer.summarize_and_plot()
