from __future__ import division
import pandas as pd
import scipy
from scipy.spatial.distance import euclidean
import time
import numpy

start = time.clock()
def readCsv(trainSet, testSet):
    colnames = ['user', 'movie', 'rating', 'time']
    train = pd.read_csv(trainSet, names=colnames, header=None)
    test = pd.read_csv(testSet, names=colnames, header=None)
    return train, test

def movie_watchers(train, test):
    movieDict = {}
    for i in test.movie:
        l1= []
        l2 = []
        d = train.loc[train.movie == i]
        l2 = d['user']
        l2 = l2.tolist()
        movieDict[i] = l2
    return movieDict

def predict_rating(test, movieDict):
    predictionList = []
    difference = 0
    for index, row in test.iterrows():
        usr = row['user']
        movie = row['movie']
        movieList = movieDict[movie]
        ratingList = [t_pivot.ix[u, movie] for u in movieList]
        mean = 0
        if len(movieList) > 0:
            mean = sum(ratingList)/len(ratingList)

        difference += abs(row['rating'] - mean)
        predictionList.append(mean)
    test['prediction'] = predictionList
    return difference

def start_process(train, test, t_pivot):
    movieDict = movie_watchers(train, test)
    difference = predict_rating(test, movieDict)
    return difference

trainFiles = ['train1.csv','train2.csv','train3.csv','train4.csv','train5.csv']
testFiles = ['test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv']
#trainFiles = ['train1.csv']
#testFiles = ['test1.csv']
k = 100
summation = 0
length = 0
for i in range(len(trainFiles)):
    train, test = readCsv(trainFiles[i], testFiles[i])
    t_pivot = train.pivot_table(index='user', columns='movie', values='rating', fill_value=0, aggfunc='first')
    summation += start_process(train, test, t_pivot)
    length += len(test.index)

MAD = summation/length
print("MAD :: ", MAD)
print time.clock() - start

