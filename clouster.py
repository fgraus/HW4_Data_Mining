from getData import getTrainingRatingMatrixAsTable
from getData import getTestMatrix
import math
import numpy as np
import pandas as pd

def main(read = False):
    dataTest = getTestMatrix()

    if not read:
        dataTrain = getTrainingRatingMatrixAsTable()

        #create columns with the centers of the clubsters
        dataTrain['center'] = np.NaN

        # number of clubsters should be totalNumber/numberNeighbords
        number_clubsters = round(dataTrain.shape[0]/25)

        data_clustered = k_means(dataTrain, number_clubsters)

        data_clustered.to_pickle('data_clouster2.pkl')
    else:
        data_clustered = pd.read_pickle('data_clouster2.pkl')

    predict(dataTest, data_clustered)
    return

def predict(dataTest, data_clustered):

    data = pd.DataFrame(columns=['values'])
    for i in range(dataTest.shape[0]):
        print('Iteracion: ' + str(i) + ' of ' + str(dataTest.shape[0]))
        element = dataTest.iloc[i]
        try:
            neigh = data_clustered[data_clustered.center == data_clustered.loc[element.userID].center]
            neigh = neigh[neigh[element.movieID] != 0]
            if neigh.empty:
                mean = 0.0
            else:
                #mean = computePrediction(data_clustered.loc[element.userID], neigh, element.movieID)
                mean = computeFasterPrediction(neigh, element.movieID)
                mean = round(mean,1)
        except KeyError:
            mean = 0.0
        data = data.append({'values' : mean}, ignore_index = True)
    data.to_pickle('data_clouster_predictions.pkl')
    return

def computeFasterPrediction(neigh, movieID):
    reviews = neigh[movieID]
    return reviews.mean()

def computePrediction(element, neighbourds, movieID):
    meanElement = element[:-1][element != 0].mean()
    num = 0
    den = 0
    for i in range(neighbourds.shape[0]):
        neigh = neighbourds.iloc[i]
        similarity = returnCosSim(element[:-1].values, neigh[:-1].values)
        meanNeigh = neigh[:-1][neigh != 0].mean()
        num += similarity*(neigh[movieID] - meanNeigh)
        den += meanNeigh
    return meanElement + (num/den)

def k_means(data, kNumber):
    centers_points = getInitialCenters(data, kNumber)
    stop = False
    iteration = 0
    while(not stop):
        data = assignPoints(data, centers_points)
        old_points = centers_points.copy()
        centers_points = computeNewCenters(data, centers_points)
        stop = iteration >= 10 or checkIfContinue(old_points, centers_points)
        iteration += 1
        print('Iteracion: ' + str(iteration))
    return data

def computeNewCenters(data, centers_points):
    for center in range(centers_points.shape[0]):
        clouster = data[data.center == center]
        for feature in range(centers_points.shape[1]-1):
            values = clouster.iloc[:,feature]
            mean = values[values!=0].mean()
            if(math.isnan(mean)):
                continue
            centers_points.at[centers_points.iloc[center].name, centers_points.columns[feature]] = mean
    return centers_points

def getInitialCenters(data, kNumber):
    return data.sample(n = kNumber)

def checkIfContinue(old_points, new_points):
    return old_points.equals(new_points)

def assignPoints(data, centers_points):
    for element in range(data.shape[0]):
        nearest_point = 0
        nearest_distance = 9999999
        for center in range(centers_points.shape[0]):
            distance = returnCosSim(data.iloc[element].values, centers_points.iloc[center].values)
            if(distance < nearest_distance):
                nearest_point = center
                nearest_distance = distance
        data.at[data.iloc[element].name, 'center'] = nearest_point
    return data

def returnCosSim(one, two):
    # eliminate the last one that have the center value
    one = one[:-1]
    two = two[:-1]

    sim = np.dot(one,two)/(np.dot(one,one)*np.dot(two,two))
    return sim

main()