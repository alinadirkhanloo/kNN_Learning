import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# import pygal
"""
Input file names:

mnist_train.csv
mnist_test.csv

"""


class KNN(object):

    def __init__(self):
        self.train_data = pd.read_csv('mnist_train.csv').values.tolist()
        self.test_data = pd.read_csv('mnist_test.csv').values.tolist()
        self.train_lable = []
        self.test_lable = []
        self.predict_test = []
        self.test_error = {1: 0, 9: 0, 19: 0, 29: 0, 39: 0, 49: 0, 59: 0, 69: 0, 79: 0, 89: 0, 99: 0}
        self.train_error = {1: 0, 9: 0, 19: 0, 29: 0, 39: 0, 49: 0, 59: 0, 69: 0, 79: 0, 89: 0, 99: 0}

    def load_train_data(self):
        for r in self.train_data:
            self.train_lable.append(r.pop(0))
        self.train_data = np.array(self.train_data)

    def load_test_data(self):
        for r in self.test_data:
            self.test_lable.append(r.pop(0))
        self.test_data = np.array(self.test_data)

    def get_distance_matrix_of_test_to_train(self):
        temp = []
        j = 0
        for test1 in self.test_data:
            i = 0
            for train1 in self.train_data:
                temp.append((np.sqrt(np.sum((train1 - test1) ** 2)), self.train_lable[i]))
                i += 1
            temp.sort()
            self.predict_test_data(temp, j)
            temp = []
            j += 1

        temp = []
        j = 0
        for test2 in self.train_data:
            i = 0
            for train2 in self.train_data:
                temp.append((np.sqrt(np.sum((train2 - test2) ** 2)), self.train_lable[i]))
                i += 1
            temp.sort()
            self.predict_train_data(temp, j)
            temp = []
            j += 1


    def predict_test_data(self, temp, test_index):
        neighbors = []
        k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        for neighbor in k:
            err_count = 0
            for i in range(len(neighbors), neighbor):
                neighbors.append(temp[i][1])
            dict = Counter(neighbors)
            dict = dict.most_common(1)[0][0]
            if not dict == self.test_lable[test_index]:
                err_count += 1
            self.test_error[neighbor] += err_count


    def predict_train_data(self, temp, train_index):
        neighbors = []
        k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        for neighbor in k:
            err_count = 0
            for i in range(len(neighbors), neighbor):
                neighbors.append(temp[i][1])
            dict = Counter(neighbors)
            dict = dict.most_common(1)[0][0]
            if not dict == self.train_lable[train_index]:
                err_count += 1
            self.train_error[neighbor] += err_count

    def plot_learning_curve(self, plot_curve1_dict,plot_curve2_dict):
        k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        plt.title("KNN", fontsize=24)
        plt.xlabel("K", fontsize=14)
        plt.ylabel("Error", fontsize=14)
        plt.plot(k, plot_curve1_dict, linewidth=5)
        plt.plot(k, plot_curve2_dict, linewidth=5)
        plt.show()

    def run(self):
        # Run your algorithm for different Ks.

        self.load_train_data()
        self.load_test_data()
        self.get_distance_matrix_of_test_to_train()
        results1 = []
        results2 = []
        for v in self.test_error.values():
            results1.append(v)
        for v in self.train_error.values():
            results2.append(v)
        self.plot_learning_curve(results1,results2)


if __name__ == "__main__":
    obj = KNN()
    obj.run()
