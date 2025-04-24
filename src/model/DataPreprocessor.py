import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from config import PROJECT_DIR


class DatasetPreprocessor:
    def __init__(self, csv_path=None, test_size=0.3, random_state=67):
        self.project_url = PROJECT_DIR
        self.csv_path = csv_path or self.project_url + '/data/final/dataset.csv'
        self.test_size = test_size
        self.random_state = random_state

        self.data = None
        self.featureset = None
        self.label = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.numerical_columns = None

    def load_and_clean(self):
        self.data = pd.read_csv(self.csv_path)
        self.data["Parkinson's Disease status"] = self.data["Parkinson's Disease status"].replace('suspectedpd', 'pd')
        self.data = self.data.dropna()
        self.data = self.data[~self.data['Device_type'].isin(['Linux Desktop', 'Desktop'])]

        race_collapse = {
            'White': "White",
            "Black or African American": "Others",
            "American Indian or Alaska Native": "Others",
            "Native Hawaiian or Other Pacific Islander": "Others",
            "Asian": "Others",
            "Prefer not to specify": "Others"
        }
        self.data['Race'] = self.data['Race'].map(race_collapse)

        gender_mapping = {'Male': 1.0, 'Female': 0.0}
        race_mapping = {'White': 0.0, "Others": 1.0}
        device_mapping = {"Windows Desktop": 0.0, "Mac Desktop": 1.0, "Linux Desktop": 0.0, "Desktop": 0.0}
        hand_mapping = {"Right": 1.0, "Left": 0.0}

        self.data['Sex_protected'] = self.data['Gender'].map(gender_mapping)
        self.data['Ethnicity_protected'] = self.data['Race'].map(race_mapping)
        self.data['Platform_protected'] = self.data['Device_type'].map(device_mapping)
        self.data['Preferred_hand_protected'] = self.data['Dominant_hand'].map(hand_mapping)

    def encode_and_split(self):
        self.featureset = self.data.drop("Parkinson's Disease status", axis=1)
        self.label = self.data["Parkinson's Disease status"]
        le = LabelEncoder()
        self.label = le.fit_transform(self.label)

        categorical_columns = self.featureset.select_dtypes(include=['object']).columns
        self.featureset = pd.get_dummies(self.featureset, columns=categorical_columns, prefix=categorical_columns).astype(float)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.featureset, self.label, test_size=self.test_size, random_state=self.random_state)

    def balance_data(self, desired_ratio=1.5):
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(self.X_train, self.y_train)

        current_non_pd_count = Counter(y_res)[0]
        target_pd_count = int(desired_ratio * current_non_pd_count)
        oversampler = RandomOverSampler(sampling_strategy={1: target_pd_count}, random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_res, y_res)

        self.X_train, self.y_train = X_resampled, y_resampled

    def scale_data(self):
        self.numerical_columns = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        scaler = MinMaxScaler()
        self.X_train[self.numerical_columns] = scaler.fit_transform(self.X_train[self.numerical_columns])
        self.X_test[self.numerical_columns] = scaler.transform(self.X_test[self.numerical_columns])

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


def get_processed_data():
    processor = DatasetPreprocessor()
    processor.load_and_clean()
    processor.encode_and_split()
    processor.balance_data()
    processor.scale_data()
    return processor.get_data()
