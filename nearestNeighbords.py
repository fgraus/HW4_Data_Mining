from getData import getTrainingRatingMatrix
from getData import getTestMatrix
import numpy as np

NUMBERNEIGHBOURDS = 10

def main():
    trainingData = getTrainingRatingMatrix()
    testData = getTestMatrix()

    userId = -1
    
    nearestNeighbords = []

    file = open('Resources/test.csv','w')

    for i in range(0,testData.shape[0]):
        
        nearestNeighbords = getNearestNeighbords(trainingData, testData.iloc[i])

        predict = returnPrediction(trainingData, nearestNeighbords, testData.iloc[i])

        file.write(str(predict) + '\n')
        print('file ' + str(i) + ' of ' + str(testData.shape[0]))
        
    file.close()
    return 0

def getNearestNeighbords(trainList, testElement):
    
    # AÑADIR TAMBIEN LOS VALORES ANTICORRELADOS, CUANDO ES MENOR QUE 0
    bestNeigbours = np.zeros((NUMBERNEIGHBOURDS,2))

    minSim = 0
    posMin = 0

    usersId = trainList['userID'].unique().tolist()

    for user in usersId:
        if(user == testElement.userID):
            continue

        userMovies = trainList[trainList['userID'] == user]

        # delete all the users that hadn't rated the movie
        if userMovies[userMovies['movieID'] == testElement.movieID].shape[0] == 0:
            continue

        sim = returnSimilarity(userMovies, trainList[trainList['userID'] == testElement.userID])
        if(sim > minSim):
            bestNeigbours[posMin][0] = sim
            bestNeigbours[posMin][1] = user

            posMin = np.unravel_index(bestNeigbours.argmin(), bestNeigbours.shape)[0]
            minSim = bestNeigbours[posMin][0]
    return bestNeigbours

def returnSimilarity(train,test):
    # PEARSON SIMILARITY
    # mean is of all the movies not only the commond ones
    meanTrain = train['rating'].mean()
    meanTest = test['rating'].mean()

    commondMovies = train[train['movieID'].isin(test['movieID'].values)]

    # if there is not commond movies the similarity between users is 0
    if commondMovies.shape[0] <= 7:
        return 0

    sumNum = 0
    sumDenTest = 0
    sumDenTrain = 0
    for movieID in commondMovies['movieID'].values:
        sumNum += (train[train['movieID'] == movieID].rating.values[0] - meanTrain)*(test[test['movieID'] == movieID].rating.values[0] - meanTest)
        sumDenTest += (train[train['movieID'] == movieID].rating.values[0] - meanTrain)**2
        sumDenTrain += (test[test['movieID'] == movieID].rating.values[0] - meanTest)**2

    pearsonSim = sumNum / (sumDenTrain**0.5 * sumDenTest**0.5)

    return pearsonSim

def returnPrediction(trainList, listNeighbours, test):

    testUserMedia = trainList[trainList['userID'] == test.userID]['rating'].mean()
    num = 0
    den = 0

    for neight in listNeighbours:
        rating = trainList[ (trainList['userID'] == neight[1]) & (trainList['movieID'] == test.movieID) ].iloc[0].rating
        num += neight[0] * (rating - trainList[trainList['userID'] == neight[1]]['rating'].mean())
        den += neight[0]
    pred = testUserMedia + (num/den)
    return round(pred,1)


main()