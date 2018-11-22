# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:13:23 2018

@author: EdvanSoares
"""

import nb_knn
filename = 'D:\\Meus Documentos\\UFPE\\Mestrado\\Semestre 2\\AM\\projeto 1\\seg_rgb.csv'
splitRatio = 0.67
dataset = nb_knn.loadCsv(filename)
trainingSet, testSet = nb_knn.splitDataset(dataset, splitRatio)
trainCV = nb_knn.cross_validation_split(trainingSet,10)
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))


accuracies = []
acuracias_final = []
acuracia_media = []
k_list = range(1,30)
best_k = None
best_accuracy = 0
k_best_dict = {}

for i in range(30):
    print("Time: "+str(i))
    for d in trainCV:
        train_cv = d[0]
        validation_cv = d[1]
        for v in validation_cv:
            for k in k_list:
                train = nb_knn.getNeighbors(train_cv, v, k)
                summaries = nb_knn.summarizeByClass(train)
                predictions = nb_knn.getPredictions(summaries, [v])
            accuracy = nb_knn.getAccuracy([v], predictions)
            accuracies.append(accuracy)
            #print('Accuracy: {0}%'.format(accuracy)
        acc_temp = nb_knn.mean(accuracies)
        if acc_temp > best_accuracy:
            print("BEST")
            k_best = k
            best_accuracy = acc_temp
        acuracias_final.append(best_accuracy)    
        k_best_dict[k_best] = best_accuracy
        k_best = 0
        best_accuracy = 0
        accuracies = []

k_final = None
import operator
k_final = max(k_best_dict.items(), key=operator.itemgetter(1))[0]


for t in testSet:
    train = nb_knn.getNeighbors(trainingSet, t, k_final)
    summaries = nb_knn.summarizeByClass(train)
    predictions = nb_knn.getPredictions(summaries, [t])
    accuracy = nb_knn.getAccuracy([t], predictions)
    accuracies.append(accuracy)
    #print('Accuracy: {0}%'.format(accuracy))

print(nb_knn.mean(accuracies))

# prepare model
#summaries = summarizeByClass(trainingSet)
# test model
#predictions = getPredictions(summaries, testSet)
#accuracy = getAccuracy(testSet, predictions)
#print('Accuracy: {0}%'.format(accuracy))

