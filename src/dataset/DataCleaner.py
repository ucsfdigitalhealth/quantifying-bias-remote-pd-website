import ast
import pandas as pd


class DataCleaner:
    def __init__(self, raw_data: pd.DataFrame):
        self.data = raw_data.copy()

    def clean_data(self):
        game_data = ['boxclicktimes', 'correctboxClicks', 'expectedKeys.round1',
                     'expectedKeys.round2', 'expectedKeys.round3', 'falseClicks',
                     'gameData.gameData', 'gameData.user','keyPresses.round1',
                     'keyPresses.round2', 'keyPresses.round3','r1points', 'pDominantTimes',
                     'pNonDominantTimes', 'qDominantTimes', 'qNonDominantTimes', 'r2points',
                     'r3points', 'rightClicks', 'timestamps.round1', 'timestamps.round2',
                     'timestamps.round3', 'wrongClicks', 'wrongboxClicks']
        self.data = self.data.dropna(subset=game_data)

    def safe_literal_eval(self, val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val
        return val

    def preprocess_columns(self):
        list_columns = ['boxclicktimes', 'expectedKeys.round1', 'expectedKeys.round2', 'expectedKeys.round3',
                        'falseClicks', 'keyPresses.round1', 'keyPresses.round2',
                        'keyPresses.round3', 'timestamps.round1', 'timestamps.round2',
                        'timestamps.round3','rightClicks', 'wrongClicks', 'therapies', 'medications']

        for column in list_columns:
            self.data[column] = self.data[column].apply(self.safe_literal_eval)

    def split_by_next(self, s: str):
        parts = s.split('NEXT')
        return parts[:-1] if parts[-1] == '' else parts

    def transform_next_columns(self):
        next_columns = ['pDominantTimes', 'pNonDominantTimes', 'qDominantTimes',
                        'qNonDominantTimes', 'r1points', 'r2points', 'r3points']

        for column in next_columns:
            self.data[column] = self.data[column].apply(self.split_by_next)

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.data
