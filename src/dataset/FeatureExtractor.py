import json
import numpy as np
import pandas as pd


class FeatureExtractor:
    def __init__(self, cleaned_data: pd.DataFrame, output_path: str):
        self.data = cleaned_data
        self.output_path = output_path
        self.features = []
        self.labels = []
        self.feature_names = [
            "Participant ID", "Parkinson's Disease status", "Age", "Gender", "Race",
            "1. Mean deviation from centerline when tracing straight line (fraction of screen height)",
            "2. Amount of time taken to trace straight line (ms)",
            "2.5. Amount of time taken to trace straight line with respect to window width (miliseconds)",
            "3. Percentage of points traced inside straight line",
            "3.5. Number of points traced inside straight line with no regard to time taken",
            "4. Amount of time taken to trace sine wave (miliseconds)",
            "4.5. Amount of time taken to trace sine wave with respect to window width (miliseconds)",
            "5. Percentage of points traced inside sine wave",
            "5.5. Number of points traced inside sine wave with no regard to time taken",
            "6. Amount of time taken to trace spiral (miliseconds)",
            "6.5. Amount of time taken to trace spiral with respect to window width (miliseconds)",
            "7. Percentage of points traced inside spiral",
            "7.5 Percentage of points traced inside spiral with no regard to time taken",
            "8. False presses when prompted with a constant key",
            "9. False presses when prompted with a semi-random(random between 2 letters) key",
            "10. False presses when prompted with a random key",
            "10.5 Total false presses from all tests",
            "10.5X Average false presses from all tests",
            "11. Average response time when prompted with a constant key (miliseconds)",
            "11.5 Sum of response times when prompted with a constant key (miliseconds)",
            "12. Average response time when prompted with a semi-random(random between 2 letters) key (miliseconds)",
            "12.5 Sum of response times when prompted with a semi-random(random between 2 letters) key (miliseconds)",
            "13. Average response time when prompted with a random key (miliseconds)",
            "13.5 Sum of response times when prompted with a random key (miliseconds)",
            "14. Number of correctly pressed keys when prompted with a constant key",
            "15. Number of correctly pressed keys when prompted with a semi-random(random between 2 letters) key",
            "16. Number of correctly pressed keys when prompted with a random key",
            "16.5. Number of correctly pressed keys when prompted with a random key with respect to average response time",
            "17. Maximum deviation from centerline when tracing straight line (percentage of screen height)",
            "17.5 Maximum deviation from centerline when tracing straight line without regard to window height (pixels)",
            "18. Net accumulated deviation from centerline when tracing straight line (percentage of screen height)",
            "19. Total accumulated deviation from centerline when tracing straight line (percentage of screen height)",
            "20. Avg of absolute values of deviation from centerline when tracing straight line (percentage of screen height)",
            "21.1 Average Time from Dominant Hands for pressing p",
            "21.2 Average Time from Non-Dominant Hands for pressing p",
            "21.3 Ratio of Average Time from Dominant Hand and Non-Dominant Hand for pressing p",
            "21.4 Total Time from Dominant Hands for pressing p",
            "21.5 Total Time from Non-Dominant Hands for pressing p",
            "22.1 Average Time from Dominant Hands for pressing q",
            "22.2 Average Time from Non-Dominant Hands for pressing q",
            "22.3 Ratio of Average Time from Dominant Hand and Non-Dominant Hand for pressing q",
            "22.4 Total Time from Dominant Hands for pressing q",
            "22.5 Total Time from Non-Dominant Hands for pressing q",
            "23.1 Average Time taken for clicking Box",
            "23.2 Total Time taken for clicking Box",
            "23.3 Standard Deviation of Time taken for clicking Box",
            "24.1 Average Reaction time for Data collected from Game",
            "24.2 Total Time for Data collected from Game",
            "24.3 Standard Deviation of Time for Data collected from Game",
            "24.4 Difference of Maximum and Minimum Reaction time for Data Collected from Game",
            "25.1 Mean Right Clicks",
            "25.2 Total Right Clicks",
            "25.3 Standard Deviation Right Clicks",
            "26.1 Mean Wrong clicks",
            "26.2 Total Wrong clicks",
            "26.3 Standard Deviation Wrong Clicks",
            "27 Participant Height",
            "28 Dominant Hand",
            "29 Device Type",
            "30 Correct number of Box Clicks"
        ]

    def analyze_point(self, points):
        counter = 0
        sum_deviation = 0
        abs_deviation = 0
        max_deviation = 0

        for point in points:
            if ('true' in point):
                counter += 1;

            p = point.replace('false', 'fal')

            if len (p.split('s')) < 4:
                continue

            deviation = float(p.split('s')[3])
            sum_deviation += deviation
            abs_deviation += abs(deviation)

            if (abs(deviation) > abs(max_deviation)):
                max_deviation = deviation

        mean_abs_deviation = abs_deviation/len(points)

        return counter, sum_deviation, abs_deviation, mean_abs_deviation, max_deviation

    def analyze_keyboard_data(self, data, expected, real):
        correct = 0
        total = 0

        for i, point in enumerate(data):
            if (expected[i] == real[i]): correct+=1
            if point == '': continue
            total += float(point)

        return correct, total

    def extract_features(self):
        self.data['false1'], self.data['false2'], self.data['false3'] = list(zip(*self.data['falseClicks']))

        for _, row in self.data.iterrows():
            port_height, port_width = row['height'], row['width']
            inside_points = 0
            feature_vector = []
            label = 1 if row['status'] == 'pd' else 0

            feature_vector.append(row['user'])
            feature_vector.append(row['status'])
            feature_vector.append(row['age'])
            feature_vector.append(row['gender'])
            feature_vector.append(row['race'])

            r1_counter, r1_sum_deviation, r1_abs_deviation, r1_mean_abs_deviation, r1_max_deviation = self.analyze_point(row['r1points'])
            r2_counter, r2_sum_deviation, r2_abs_deviation, r2_mean_abs_deviation, r2_max_deviation = self.analyze_point(row['r2points'])
            r3_counter, r3_sum_deviation, r3_abs_deviation, r3_mean_abs_deviation, r3_max_deviation = self.analyze_point(row['r3points'])

            k1_correct, k1_total = self.analyze_keyboard_data(row['timestamps.round1'], row['expectedKeys.round1'], row['keyPresses.round1'])
            k2_correct, k2_total = self.analyze_keyboard_data(row['timestamps.round2'], row['expectedKeys.round2'], row['keyPresses.round2'])
            k3_correct, k3_total = self.analyze_keyboard_data(row['timestamps.round3'], row['expectedKeys.round3'], row['keyPresses.round3'])

            pDominantTimes = [int(num) for num in row['pDominantTimes']]
            pNonDominantTimes = [int(num) for num in row['pNonDominantTimes']]
            qDominantTimes = [int(num) for num in row['qDominantTimes']]
            qNonDominantTimes = [int(num) for num in row['qNonDominantTimes']]
            gameData = json.loads(row['gameData.gameData'])
            reaction_time_values = [entry["reactionTime"] for entry in gameData]

            feature_vector.append((r1_sum_deviation / len(row['r1points'])) / float(port_height)) #feature 1
            feature_vector.append(len(row['r1points']) * 500) #feature 2
            feature_vector.append((len(row['r1points'])/float(port_width))*500) #feature 2.5
            feature_vector.append(r1_counter / len(row['r1points'])) #feature 3
            feature_vector.append(r1_counter) #feature 3.5
            feature_vector.append(len(row['r2points']) * 500) #feature 4
            feature_vector.append((len(row['r2points'])/float(port_width))*500) #feature 4.5
            feature_vector.append(r2_counter / len(row['r2points'])) #feature 5
            feature_vector.append(r2_counter) #feature 5.5
            feature_vector.append(len(row['r3points']) * 500) #feature 6
            feature_vector.append((len(row['r3points'])/float(port_width))*500) #feature 6.5
            feature_vector.append(r3_counter / len(row['r3points'])) #feature 7
            feature_vector.append(r3_counter) #feature 7.5
            feature_vector.append(int(row['false1'])) #feature 8
            feature_vector.append(int(row['false2'])) #feature 9
            feature_vector.append(int(row['false3'])) #feature 10
            feature_vector.append(int(row['false1']) + int(row['false2']) + int(row['false3'])) #feature 10.5
            feature_vector.append(np.mean([int(row['false1']), int(row['false2']), int(row['false3'])])) #feature 10.5X
            feature_vector.append(k1_total/len(row['timestamps.round1'])) #feature 11
            feature_vector.append(k1_total) #feature 11.5
            feature_vector.append(k2_total/len(row['timestamps.round2'])) #feature 12
            feature_vector.append(k2_total) #feature 12.5
            feature_vector.append(k3_total/len(row['timestamps.round3'])) #feature 13
            feature_vector.append(k3_total) #feature 13.5
            feature_vector.append(k1_correct) #feature 14
            feature_vector.append(k2_correct) #feature 15
            feature_vector.append(k3_correct) #feature 16
            feature_vector.append(k3_correct/(k3_total/len(row['timestamps.round3']))) #feature 16.5
            feature_vector.append(r1_max_deviation / float(port_height)) #feature 17
            feature_vector.append(r1_max_deviation) #feature 17.5
            feature_vector.append(r1_sum_deviation / float(port_height)) #feature 18
            feature_vector.append(r1_abs_deviation/float(port_height)) #feature 19
            feature_vector.append(r1_mean_abs_deviation/float(port_height)) #feature 20
            feature_vector.append(np.mean(pDominantTimes)) #feature 21.1
            feature_vector.append(np.mean(pNonDominantTimes)) #feature 21.2
            feature_vector.append(np.mean(pDominantTimes)/np.mean(pNonDominantTimes)) #feature 21.3
            feature_vector.append(np.sum(pDominantTimes)) #feature 21.4
            feature_vector.append(np.sum(pNonDominantTimes)) #feature 21.5
            feature_vector.append(np.mean(qDominantTimes)) #feature 22.1
            feature_vector.append(np.mean(qNonDominantTimes)) #feature 22.2
            feature_vector.append(np.mean(qDominantTimes)/np.mean(qNonDominantTimes)) #feature 22.3
            feature_vector.append(np.sum(qDominantTimes)) #feature 22.4
            feature_vector.append(np.sum(qNonDominantTimes)) #feature 22.5
            feature_vector.append(np.mean(row['boxclicktimes'])) #feature 23.1
            feature_vector.append(np.sum(row['boxclicktimes'])) #feature 23.2
            feature_vector.append(np.std(row['boxclicktimes'])) #feature 23.3
            feature_vector.append(np.mean(reaction_time_values)) #feature 24.1
            feature_vector.append(np.sum(reaction_time_values)) #feature 24.2
            feature_vector.append(np.std(reaction_time_values)) #feature 24.3
            feature_vector.append(np.max(reaction_time_values) - np.min(reaction_time_values)) #feature 24.4
            feature_vector.append(np.mean(row['rightClicks'])) #feature 25.1
            feature_vector.append(np.sum(row['rightClicks'])) #feature 25.2
            feature_vector.append(np.std(row['rightClicks'])) #feature 25.3
            feature_vector.append(np.mean(row['wrongClicks'])) #feature 26.1
            feature_vector.append(np.sum(row['wrongClicks'])) #feature 26.2
            feature_vector.append(np.std(row['wrongClicks'])) #feature 26.3
            feature_vector.append(row['Participant_height']) #feature 27
            feature_vector.append(row['dominantHand']) #feature 28
            feature_vector.append(row['deviceType']) #feature 29
            feature_vector.append(row['correctboxClicks']) #feature 30

            self.features.append(feature_vector)
            self.labels.append(label)

    def save_dataset(self):
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df.to_csv(self.output_path, index=False)
        print(f"Dataset saved at {self.output_path}")

    def run_pipeline(self):
        self.extract_features()
        self.save_dataset()