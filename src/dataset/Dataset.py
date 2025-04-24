import os
import pandas as pd
import numpy as np
from src.dataset.FeatureExtractor import FeatureExtractor
from src.dataset.DataCleaner import DataCleaner


class Dataset:
    def __init__(self, input_path: str, output_path: str):
        self.output_path = output_path
        raw_data = pd.read_csv(input_path)
        self.data_cleaner = DataCleaner(raw_data)
        cleaned_data = self.clean_data()
        self.feature_extractor = FeatureExtractor(cleaned_data, output_path)

    def clean_data(self):
        self.data_cleaner.clean_data()
        self.data_cleaner.preprocess_columns()
        self.data_cleaner.transform_next_columns()
        return self.data_cleaner.get_cleaned_data()

    def process_and_save(self):
        self.feature_extractor.run_pipeline()

    def get_extracted_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_extractor.features)

    def apply_feature_name_shortening(self, feature_name_mapping):
        extracted_features = pd.read_csv(self.output_path)
        extracted_features.drop(columns=['Participant ID'], inplace=True)
        extracted_features.rename(columns=feature_name_mapping, inplace=True)

        base, ext = os.path.splitext(self.output_path)
        self.shorten_output_path = f"{base}_shorten{ext}"
        extracted_features.to_csv(self.shorten_output_path, index=False)

    def apply_feature_engineering(self, final_output_path:str):
        processed_dataset = pd.read_csv(self.shorten_output_path)
        processed_dataset["Parkinson's Disease status"] = processed_dataset["Parkinson's Disease status"].replace('suspectedpd', 'pd')

        processed_dataset['false_presses_over_total_key_presses_constant_key'] = processed_dataset['False_pck'] / (processed_dataset['False_pck'] + processed_dataset['Num_crpck'])
        processed_dataset['false_presses_over_total_key_presses_semi-random_key'] = processed_dataset['False_psr'] / (processed_dataset['False_psr'] + processed_dataset['Num_crpsr'])
        processed_dataset['false_presses_over_total_key_presses_random_key'] = processed_dataset['False_prk'] / (processed_dataset['False_prk'] + processed_dataset['Num_crprk'])


        processed_dataset['mean_false_presses_over_total_key_presses'] = np.mean([
            processed_dataset['false_presses_over_total_key_presses_constant_key'],
            processed_dataset['false_presses_over_total_key_presses_semi-random_key'],
            processed_dataset['false_presses_over_total_key_presses_random_key']
            ], axis=0)

        processed_dataset ['std of the pfalse_presses_over_total_key_presses'] = np.std([
            processed_dataset['false_presses_over_total_key_presses_constant_key'],
            processed_dataset['false_presses_over_total_key_presses_semi-random_key'],
            processed_dataset['false_presses_over_total_key_presses_random_key']
            ], axis =0)


        processed_dataset['mean_of_total_amount_of_time_taken_to_trace_lines'] = np.mean([
            processed_dataset['A_tcl'],
            processed_dataset['A_tsw'],
            processed_dataset['A_tts']
            ], axis=0)


        processed_dataset['mean_of_total_amount_of_time_taken_to_trace_lines_with_respect_to_window_width'] = np.mean([
            processed_dataset['A_tslww'],
            processed_dataset['A_tww'],
            processed_dataset['A_spww']
            ], axis=0)



        processed_dataset['mean_of_total_Number_of_points_to_traced_inside_line_with_no_regard_to_time_taken'] = np.mean([
            processed_dataset['N_tislnt'],
            processed_dataset['Num_iswtnt'],
            processed_dataset['P_spnt']
            ], axis=0)


        processed_dataset['Interaction_Time_Straight_False_Presses']=processed_dataset['A_tcl'] * processed_dataset['False_pck']
        processed_dataset['Interaction_Time_Sine_Wave_False_presses'] =processed_dataset['A_tsw']* processed_dataset['False_pck']
        processed_dataset['Interaction_Time_Spiral_False_presses'] =processed_dataset['A_tts']* processed_dataset['False_pck']

        processed_dataset['Number of correctly pressed keys when prompted with a constant key with respect to average response time'] = processed_dataset['Num_crpck']/processed_dataset['Avg_pck']
        processed_dataset['Number of correctly pressed keys when prompted with a semi-random key with respect to average response time'] = processed_dataset['Num_crpsr']/processed_dataset['Avg_psr']


        processed_dataset['Ratio_Points_Sine_Straight'] = processed_dataset['P_ptsw']/processed_dataset['P_isl']
        processed_dataset['Ratio_Points_Spiral_Straight'] = processed_dataset['P_insp']/processed_dataset['P_isl']
        processed_dataset['Ratio_Points_Sine_Spiral'] = processed_dataset['P_ptsw']/processed_dataset['P_insp']


        processed_dataset.to_csv(final_output_path, index=False)
        print(f"Feature Engineered Dataset saved at {final_output_path}")