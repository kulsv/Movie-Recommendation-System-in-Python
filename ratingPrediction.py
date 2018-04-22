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

def find_neighbors(t_pivot, distFunction):
    ary = scipy.spatial.distance.cdist(t_pivot.iloc[:,1:],t_pivot.iloc[:,1:],metric=distFunction)
    distance = pd.DataFrame(ary)
    userDict = {}
    for index, row in distance.iterrows():
        ll = numpy.asarray(row).argsort().tolist()
        ll.remove(index)
        ll = map(lambda x: x+1, ll)
        userDict[index+1] = ll
    return userDict

def movie_watchers(train, test):
    movieDict = {}
    movieList = train.movie.unique()
    for i in movieList:
        if i not in movieDict:
            l2 = []
            d = train.loc[train.movie == i]
            l2 = d['user']
            l2 = l2.tolist()
            movieDict[i] = l2
    return movieDict

def predict_rating(test, movieDict, userDict):
    predictionList = []
    difference = 0
    for index, row in test.iterrows():
        usr = row['user']
        movie = row['movie']
        neighborList = userDict[usr]
        if movie not in movieDict:
            difference += 0
            mean = 0
            predictionList.append(mean)
            continue
        movieList = movieDict[movie]
        s = set(movieList)
        commonUsers = [x for x in neighborList if x in s]
        ratingList = [t_pivot.ix[u, movie] for u in commonUsers]   #check
        if len(ratingList) >= k:
            ratingList = numpy.array(ratingList)
            mean = numpy.mean(ratingList[:k])
            difference += abs(row['rating']-mean)
            predictionList.append(mean)
        else:
            if len(ratingList) > 0:
                ratingList = numpy.array(ratingList)
                mean = numpy.mean(ratingList[:len(ratingList)])
                difference += abs(row['rating'] - mean)
                predictionList.append(mean)
            else:
                mean = 0
                ratingList = []
                ratingList = [t_pivot.ix[u, movie] for u in movieList]
                if len(ratingList) < 1:
                    mean = 0
                else:
                    mean = sum(ratingList)/len(ratingList)
                difference += abs(row['rating'] - mean)
                predictionList.append(mean)
    test['prediction'] = predictionList
    return difference

def start_process(train, test, t_pivot, distFunction):
    userDict = find_neighbors(t_pivot, distFunction)
    movieDict = movie_watchers(train, test)
    difference = predict_rating(test, movieDict, userDict)
    return difference

trainFiles = ['train1.csv','train2.csv','train3.csv','train4.csv','train5.csv']
testFiles = ['test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv']
distFuctions = ['euclidean','cosine','cityblock']

k = 100

length = 0
for j in range(len(distFuctions)):
    summation = 0
    length = 0
    for i in range(len(trainFiles)):
        train, test = readCsv(trainFiles[i], testFiles[i])
        t_pivot = train.pivot_table(index='user', columns='movie', values='rating', fill_value=0, aggfunc='first')
        summation += start_process(train, test, t_pivot, distFuctions[j])
        length += len(test.index)
    MAD = summation/length
    print("MAD for ",distFuctions[j], " is :: ", MAD)

print time.clock() - start