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

    colnames1 = ['user', 'age', 'gender', 'job', 'zip']
    df_gender = pd.read_csv('u_user.csv', header=None, names=colnames1)

    df_gender = df_gender.drop('age', axis=1)
    df_gender = df_gender.drop('job', axis=1)
    df_gender = df_gender.drop('zip', axis=1)
    #print("df_gender is:", df_gender)

    final_train = train.merge(df_gender, on='user', how='left')
    test = pd.read_csv(testSet, names=colnames, header=None)
    return final_train, test

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
    #testmovie = test.movie.unique()
    #print("movielist :: ", movieList)
    #print("length :: ", len(movieList))
    #print("test movies :: ", testmovie)
    #print("length test :: ", len(testmovie))
    #lll = set(movieList).intersection(set(testmovie))
    #print("intersection :: ", lll)
    #print("length of intersection :: ", len(lll))
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
        #if usr == 671:
         #   print("here!!")
          #  print("userdict :: ", userDict)
        movie = row['movie']

        if movie not in movieDict:
            difference += 0
            mean = 0
            predictionList.append(mean)
            continue
        movieList = movieDict[movie]
        if usr not in userDict:
            ratingList = [t_pivot.ix[u, movie] for u in movieList]
            if len(ratingList) < 1:
                mean = 0
            else:
                mean = sum(ratingList) / len(ratingList)
            difference += abs(row['rating'] - mean)
            predictionList.append(mean)
            continue
        #if usr not in userDict:
        neighborList = userDict[usr]

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

def start_process(train, test, userDict):
    movieDict = movie_watchers(train, test)
    difference = predict_rating(test, movieDict, userDict)
    return difference

trainFiles = ['train1.csv','train2.csv','train3.csv','train4.csv','train5.csv']
testFiles = ['test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv']
distFuctions = ['euclidean','cosine','cityblock']
#distFuctions = ['euclidean']
#trainFiles = ['train3.csv']
#testFiles = ['test3.csv']
k = 10

length = 0
for j in range(len(distFuctions)):
    summation = 0
    length = 0
    for i in range(len(trainFiles)):
        train, test = readCsv(trainFiles[i], testFiles[i])
        male_df = train.loc[train.gender == 'M']
        female_df = train.loc[train.gender == 'F']
        male_pivot = male_df.pivot_table(index='user', columns='movie', values='rating', fill_value=0, aggfunc='first')
        female_pivot = female_df.pivot_table(index='user', columns='movie', values='rating', fill_value=0, aggfunc='first')
        t_pivot = train.pivot_table(index='user', columns='movie', values='rating', fill_value=0, aggfunc='first')
        male_dict = find_neighbors(male_pivot, distFuctions[j])
        female_dict = find_neighbors(female_pivot, distFuctions[j])
        userDict = {}
        userDict.update(male_dict)
        userDict.update(female_dict)

        summation += start_process(train, test, userDict)
        length += len(test.index)
    MAD = summation/length
    print("MAD for ",distFuctions[j], " is :: ", MAD)

print time.clock() - start