# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:29:51 2018

@author: Delgado
"""
import nb


filename = 'D:\\Meus Documentos\\UFPE\\Mestrado\\Semestre 2\\AM\\projeto 1\\seg.csv'
splitRatio = 0.8
dataset = nb.loadCsv(filename)
datasetSplit = nb.cross_validation_split(dataset,folds=10)
accuracies = []
acuracia_media = []
for i in range(30):
    for d in datasetSplit:
        train = d[0]
        test = d[1]
        
        # prepare model
        summaries = nb.summarizeByClass(train)
        # test model
        results = nb.getPredictions(summaries, test)
        predictions = results[0] 
        probs = results[1]
        accuracy = nb.getAccuracy(test, predictions)
        accuracies.append(accuracy)
        print('Accuracy: {0}%'.format(accuracy))
    acuracia_media.append(sum(accuracies)/len(accuracies))
print('Final Accuracy : {0}%'.format(sum(acuracia_media)/30))

#trainingSet, testSet = nb.splitDataset(dataset, splitRatio)
#print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
## prepare model
#summaries = nb.summarizeByClass(trainingSet)
## test model
#results = nb.getPredictions(summaries, testSet)
#predictions = results[0] 
#probs = results[1]
#accuracy = nb.getAccuracy(testSet, predictions)
#print('Accuracy: {0}%'.format(accuracy))

