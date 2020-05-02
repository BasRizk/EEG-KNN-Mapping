# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:23:54 2020

@author: Ibram Medhat & Basem Rizk
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

raw_training_angle_array = np.genfromtxt("Angle_Training.txt")
raw_testing_angle_array = np.genfromtxt("Angle_Testing.txt")
training_spike_trains = np.transpose(np.genfromtxt("Training_SpikeTrains.txt"))
testing_spike_trains = np.transpose(np.genfromtxt("Testing_SpikeTrains.txt"))


training_angle_array = np.where((raw_training_angle_array < 90.0) & (raw_training_angle_array >= 0.0), 1, raw_training_angle_array)
training_angle_array = np.where((raw_training_angle_array < 180.0) & (raw_training_angle_array >= 90.0), 2, training_angle_array)
training_angle_array = np.where((raw_training_angle_array < 270.0) & (raw_training_angle_array >= 180.0), 3, training_angle_array)
training_angle_array = np.where((raw_training_angle_array <= 360.0) & (raw_training_angle_array >= 270.0), 4, training_angle_array)

testing_angle_array = np.where((raw_testing_angle_array < 90.0) & (raw_testing_angle_array >= 0.0), 1, raw_testing_angle_array)
testing_angle_array = np.where((raw_testing_angle_array < 180.0) & (raw_testing_angle_array >= 90.0), 2, testing_angle_array)
testing_angle_array = np.where((raw_testing_angle_array < 270.0) & (raw_testing_angle_array >= 180.0), 3, testing_angle_array)
testing_angle_array = np.where((raw_testing_angle_array <= 360.0) & (raw_testing_angle_array >= 270.0), 4, testing_angle_array)

knn_scores = []
different_k_values = np.arange(1, 301)

for k in range(1, 301) :
    spikes_classifier = KNN(n_neighbors=k)
    spikes_classifier.fit(training_spike_trains, training_angle_array)
    current_score = 100 * spikes_classifier.score(testing_spike_trains, testing_angle_array)
    knn_scores.append(current_score)
    print(current_score)

plt.figure(1)
plt.plot(different_k_values, knn_scores)
plt.title("KNN Acurracies")

plt.savefig("KNN Accuracies.png", progressive=True)



