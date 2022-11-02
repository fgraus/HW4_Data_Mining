from getData import getTrainingRatingMatrixAsTable
from getData import getTestMatrix
import numpy as np

NUMBERNEIGHBOURDS = 10

def main():
    dataTrain = getTrainingRatingMatrixAsTable()
    dataTest = getTestMatrix()

    for i in range(dataTest.shape[0]):

        nearestNeighbords = getNearestNeighbords(dataTrain, dataTest.iloc[i])

        print('file ' + str(i) + ' of ' + str(dataTest.shape[0]))
        #predict = returnPrediction()

        #writte

    return 0

def getNearestNeighbords(dataTrain, testElement):
    
    bestNeigbours = np.zeros((NUMBERNEIGHBOURDS,2))

    minSim = 0
    posMin = 0

    testRatings = dataTrain.loc[testElement.userID]

    for i in range(dataTrain.shape[0]):
        candidate = dataTrain.iloc[i]

        # if have not rated the film we skip it
        # if it is the same user as candidate skip it
        if candidate[testElement.movieID] == 0.0 or candidate.name == testRatings.name:
            continue

        cosSim = returnCosSim(candidate.values, testRatings.values)
        if(cosSim > minSim):
            bestNeigbours[posMin][0] = cosSim
            bestNeigbours[posMin][1] = candidate.name

            posMin = np.unravel_index(bestNeigbours.argmin(), bestNeigbours.shape)[0]
            minSim = bestNeigbours[posMin][0]
    return 0

def returnCosSim(one, two):
    sim = np.dot(one,two)/(np.dot(one,one)*np.dot(two,two))
    return sim
main()