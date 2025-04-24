import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.metrics import accuracy_score, f1_score, recall_score, RocCurveDisplay
from src.model.DataPreprocessor import get_processed_data
from config import PROJECT_DIR


from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class TestRunner:
    def __init__(self, dataset_path, weights_dir):
        self.dataset_path = dataset_path
        self.weights_dir = weights_dir
        self.models = {}
        self.X_test = None
        self.y_test = None

    def prepare_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = get_processed_data()

    def load_models(self):
        self.models = {
            "Random Forest": RandomForestClassifier(max_depth= 5, random_state=97, max_features = 'log2', min_samples_leaf= 1, min_samples_split= 5, n_estimators=100, class_weight='balanced'),
            "LightGBM": LGBMClassifier(class_weight='balanced'),
            "XGBoost": XGBClassifier(n_estimators=200, random_state=89, objective='binary:logistic', gamma= 0.2, learning_rate= 0.01, max_depth=3, min_child_weight=1, reg_alpha=0, reg_lambda=0.1),
        }

    def train_models(self):
        for name, _ in self.models.items():
            self.models[name].fit(self.X_train, self.y_train)

    def evaluate_model(self, model, name):
        y_pred = model.predict(self.X_test)
        print(f"{name}")
        print(f"Accuracy Score: {accuracy_score(self.y_test, y_pred):.2f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.2f}")
        print(f"Sensitivity (Recall for PD): {recall_score(self.y_test, y_pred, pos_label=1):.2f}")
        print(f"Specificity (Recall for Non-PD):{recall_score(self.y_test, y_pred, pos_label=0):.2f}")

    def plot_roc_curves(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, model in self.models.items():
            RocCurveDisplay.from_estimator(model, self.X_test, self.y_test, ax=ax, name=name)
        plt.title("ROC Curve Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIR, "analysis/testing.png"))
        plt.show()
        print("📁 Saved plot to analysis/testing.png")

    def run(self):
        self.prepare_data()
        self.load_models()
        self.train_models()
        for name, model in self.models.items():
            self.evaluate_model(model, name)
        self.plot_roc_curves()


def test_model():
    runner = TestRunner(
        dataset_path= PROJECT_DIR + "/data/final/dataset.csv",
        weights_dir= PROJECT_DIR + "/weights"
    )
    runner.run()
