import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split


class DataProccess:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

        self.feat_raw = self.data['quizzes']
        self.label_raw = self.data['solutions']

        self.feat = []
        self.label = []

    def get_data(self, split=0.5, random_state=42):

        print('Initializing 1M sudoku games..\nPlease wait.')

        for i in self.feat_raw:
            x = np.array([int(j) for j in i]).reshape((9, 9, 1))
            self.feat.append(x)

        self.feat = np.array(self.feat)

        self.feat = np.array(self.feat)
        self.feat = self.feat / 9
        self.feat -= .5

        for i in self.label_raw:
            x = np.array([int(j) for j in i]).reshape((81, 1)) - 1
            self.label.append(x)

        self.label = np.array(self.label)

        del self.label_raw
        del self.feat_raw

        x_train, x_test, y_train, y_test = train_test_split(self.feat, self.label,
                                                            test_size=split,
                                                            random_state=random_state)

        print('Initialization finished.')

        return x_train, x_test, y_train, y_test

    def write_new_csv(self):

        with open("sudoku_new.csv", mode='w', newline='') as w_csv:

            fieldnames = ['quizz', 'solution']

            writer = csv.DictWriter(w_csv, fieldnames=fieldnames, delimiter=',')

            writer.writeheader()
            for i, j in zip(self.feat_raw, self.label_raw):
                writer.writerow({"quizz": str('\'') + str(i), "solution": str('\'') + str(j)})

    def get_features(self):
        return self.feat

    def get_labels(self):
        return self.label

    @staticmethod
    def norm(a):
        return (a / 9) - .5

    @staticmethod
    def denorm(a):
        return (a + .5) * 9
