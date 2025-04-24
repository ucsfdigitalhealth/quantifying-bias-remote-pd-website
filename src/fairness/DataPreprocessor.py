import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from config import PROJECT_DIR


class DataPreprocessor:
    def __init__(self, csv_path=None, random_state=67):
        self.csv_path = csv_path or PROJECT_DIR + '/data/final/dataset.csv'
        self.random_state = random_state
        self.data = None
        self.features = None
        self.labels = None

    def load_and_clean(self):
        data = pd.read_csv(self.csv_path)
        data["Parkinson's Disease status"] = data["Parkinson's Disease status"].replace('suspectedpd', 'pd')
        data = data.dropna()
        data = data[~data['Device_type'].isin(['Linux Desktop', 'Desktop'])]

        race_mapping = {'White': "White", 
                        "Black or African American": "Others",
                        "American Indian or Alaska Native": "Others"}
        
        data.loc[:, 'Race'] = data['Race'].map(race_mapping)

        gender_mapping = {'Male': 1.0, 'Female': 0.0}
        race_mapping = {'White': 1.0, "Others": 0.0}
        device_mapping = {"Windows Desktop": 0.0, "Mac Desktop": 1.0, "Linux Desktop": 0.0, "Desktop": 0.0}
        dominantHand_mapping = {"Right": 1.0, "Left": 0.0}

        data.loc[:, 'Sex_protected'] = data['Gender'].map(gender_mapping)
        data.loc[:, 'Ethnicity_protected'] = data['Race'].map(race_mapping)
        data.loc[:, 'Platform_protected'] = data['Device_type'].map(device_mapping)
        data.loc[:, 'Preferred_hand_protected'] = data['Dominant_hand'].map(dominantHand_mapping)

        self.data = data.dropna().reset_index(drop=True)
    
    def upsample_race(self):
        white_majority = self.data[(self.data['Race'] == 'White') & (self.data["Parkinson's Disease status"] == 'nonpd')]
        non_white_minority = self.data[(self.data['Race'] != 'White') & (self.data["Parkinson's Disease status"] == 'nonpd')]

        non_white_minority_upsampled = resample(non_white_minority,
                                                replace=True,
                                                n_samples=len(white_majority),
                                                random_state=42)

        balanced_data = pd.concat([white_majority, non_white_minority_upsampled])
        remaining_data = self.data[~((self.data['Race'] != 'White') & (self.data["Parkinson's Disease status"] == 'nonpd')) & ~((self.data['Race'] == 'White') & (self.data["Parkinson's Disease status"] == 'nonpd'))]

        self.data = pd.concat([balanced_data, remaining_data])

    def encode(self):
        self.featureset = self.data.drop("Parkinson's Disease status", axis=1)
        self.label = self.data["Parkinson's Disease status"]
        le = LabelEncoder()
        self.label = le.fit_transform(self.label)

        categorical_columns = self.featureset.select_dtypes(include=['object']).columns
        self.featureset = pd.get_dummies(self.featureset, columns=categorical_columns, prefix=categorical_columns).astype(float)

    def balance_data(self, desired_ratio=1.5):
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(self.featureset, self.label)

        current_non_pd_count = Counter(y_res)[0]
        target_pd_count = int(desired_ratio * current_non_pd_count)
        oversampler = RandomOverSampler(sampling_strategy={1: target_pd_count}, random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_res, y_res)

        self.X_train, self.y_train = X_resampled, y_resampled

    def scale_data(self):
        self.numerical_columns = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        scaler = MinMaxScaler()
        self.X_train[self.numerical_columns] = scaler.fit_transform(self.X_train[self.numerical_columns])

    def get_data(self):
        return self.X_train, self.y_train
    

def get_processed_data(race_upsample=False):
    processor = DataPreprocessor()
    processor.load_and_clean()
    if race_upsample:
        processor.upsample_race()
    processor.encode()
    processor.balance_data()
    processor.scale_data()
    return processor.get_data()