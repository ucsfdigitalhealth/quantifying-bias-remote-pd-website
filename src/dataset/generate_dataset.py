from src.dataset.Dataset import Dataset
from config import PROJECT_DIR as PROJECT_URL


feature_name_mapping = {
    "Parkinson's Disease status": "Parkinson's Disease status",
    "Age": "Age",
    "Gender": "Gender",
    "Race": "Race",
    "1. Mean deviation from centerline when tracing straight line (fraction of screen height)": "M_deviation",
    "2. Amount of time taken to trace straight line (ms)": "A_tcl",
    "2.5. Amount of time taken to trace straight line with respect to window width (miliseconds)": "A_tslww",
    "3. Percentage of points traced inside straight line": "P_isl",
    "3.5. Number of points traced inside straight line with no regard to time taken": "N_tislnt",
    "4. Amount of time taken to trace sine wave (miliseconds)": "A_tsw",
    "4.5. Amount of time taken to trace sine wave with respect to window width (miliseconds)": "A_tww",
    "5. Percentage of points traced inside sine wave": "P_ptsw",
    "5.5. Number of points traced inside sine wave with no regard to time taken": "Num_iswtnt",
    "6. Amount of time taken to trace spiral (miliseconds)": "A_tts",
    "6.5. Amount of time taken to trace spiral with respect to window width (miliseconds)": "A_spww",
    "7. Percentage of points traced inside spiral": "P_insp",
    "7.5 Percentage of points traced inside spiral with no regard to time taken": "P_spnt",
    "8. False presses when prompted with a constant key": "False_pck",
    "9. False presses when prompted with a semi-random(random between 2 letters) key": "False_psr",
    "10. False presses when prompted with a random key": "False_prk",
    "10.5 Total false presses from all tests": "Tof_pa",
    "10.5X Average false presses from all tests": "Avg_fpa",
    "11. Average response time when prompted with a constant key (miliseconds)": "Avg_pck",
    "11.5 Sum of response times when prompted with a constant key (miliseconds)": "Sum_pck",
    "12. Average response time when prompted with a semi-random(random between 2 letters) key (miliseconds)": "Avg_psr",
    "12.5 Sum of response times when prompted with a semi-random(random between 2 letters) key (miliseconds)": "Sum_psr",
    "13. Average response time when prompted with a random key (miliseconds)": "Avg_prk",
    "13.5 Sum of response times when prompted with a random key (miliseconds)": "Sum_prk",
    "14. Number of correctly pressed keys when prompted with a constant key": "Num_crpck",
    "15. Number of correctly pressed keys when prompted with a semi-random(random between 2 letters) key": "Num_crpsr",
    "16. Number of correctly pressed keys when prompted with a random key": "Num_crprk",
    "16.5. Number of correctly pressed keys when prompted with a random key with respect to average response time": "Num_crprk_av",
    "17. Maximum deviation from centerline when tracing straight line (percentage of screen height)": "Maxdev_cltsl",
    "17.5 Maximum deviation from centerline when tracing straight line without regard to window height (pixels)": "Maxdev_cltslw",
    "18. Net accumulated deviation from centerline when tracing straight line (percentage of screen height)": "Net_devsl",
    "19. Total accumulated deviation from centerline when tracing straight line (percentage of screen height)": "T_devsl",
    "20. Avg of absolute values of deviation from centerline when tracing straight line (percentage of screen height)": "Avg_sl",
    "21.1 Average Time from Dominant Hands for pressing p": "Avg_dhpp",
    "21.2 Average Time from Non-Dominant Hands for pressing p": "Avg_ndhpp",
    "21.3 Ratio of Average Time from Dominant Hand and Non-Dominant Hand for pressing p": "Ratio_dndhpp",
    "21.4 Total Time from Dominant Hands for pressing p": "Total_dhpp",
    "21.5 Total Time from Non-Dominant Hands for pressing p": "Total_ndhpp",
    "22.1 Average Time from Dominant Hands for pressing q": "Avg_dhpq",
    "22.2 Average Time from Non-Dominant Hands for pressing q": "Avg_ndhpq",
    "22.3 Ratio of Average Time from Dominant Hand and Non-Dominant Hand for pressing q": "Ratio_dndhpq",
    "22.4 Total Time from Dominant Hands for pressing q": "Total_dhpq",
    "22.5 Total Time from Non-Dominant Hands for pressing q": "Total_ndhpq",
    "23.1 Average Time taken for clicking Box": "Avg_ttcb",
    "23.2 Total Time taken for clicking Box": "Total_ttcb",
    "23.3 Standard Deviation of Time taken for clicking Box": "Std_ttcb",
    "24.1 Average Reaction time for Data collected from Game": "Avg_reaction",
    "24.2 Total Time for Data collected from Game": "Totaltime_game",
    "24.3 Standard Deviation of Time for Data collected from Game": "Std_game",
    "24.4 Difference of Maximum and Minimum Reaction time for Data Collected from Game": "Maxmin_reaction",
    "25.1 Mean Right Clicks": "Mean_rc",
    "25.2 Total Right Clicks": "Total_rc",
    "25.3 Standard Deviation Right Clicks": "Std_rc",
    "26.1 Mean Wrong clicks": "Mean_wc",
    "26.2 Total Wrong clicks": "Total_wc",
    "26.3 Standard Deviation Wrong Clicks": "Std_wc",
    "27 Participant Height": "Paricipant_height",
    "28 Dominant Hand": "Dominant_hand",
    "29 Device Type": "Device_type",
    "30 Correct number of Box Clicks": "Correct_boxclick"
}

def generate_dataset():
    raw_dataset = Dataset(PROJECT_URL + '/data/raw/dataset.csv', PROJECT_URL + '/data/interim/dataset.csv')
    raw_dataset.process_and_save()
    raw_dataset.apply_feature_name_shortening(feature_name_mapping)
    raw_dataset.apply_feature_engineering(PROJECT_URL + '/data/final/dataset.csv')


if __name__ == "__main__":
    generate_dataset()