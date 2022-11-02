import math
import numpy as np
import pandas as pd
from getData import getData

NUMBERNEIGHBOURDS = 1000

def returnModel(read = True):
    
    rating, test, matrix = getData()
    if not read:

        matrix['neighbourds'] = None
        for i in range(matrix.shape[0]):
            
            nearestNeighbords = getNearestNeighbords(matrix, matrix.iloc[i])
            matrix.at[matrix.iloc[i].name, 'neighbourds'] = nearestNeighbords
            
            print('file ' + str(i) + ' of ' + str(matrix.shape[0]))
        matrix.to_pickle('data_contente_based_with_neighbourds.pkl')
        return
    else:
        matrix = pd.read_pickle('data_contente_based_with_neighbourds.pkl')
    matrix = matrix.drop(['actor', 'director', 'tags', 'genre'], axis=1)
    predict(rating, test, matrix)
    return matrix

def predict(rating, dataTest, data_clustered):

    data = pd.DataFrame(columns=['values'])
    for i in range(dataTest.shape[0]):
        print('Iteracion: ' + str(i) + ' of ' + str(dataTest.shape[0]))
        element = dataTest.iloc[i]
        try:
            mean = computePrediction(data_clustered.loc[element.movieID][0], rating, element.userID)
        except KeyError:
            mean = 0.0
        mean = round(mean,1)
        data = data.append({'values' : mean}, ignore_index = True)
    data.to_pickle('data_contente_based_predictions.pkl')
    return

def computePrediction(neighbourds, ratings, user):
    nearestMovies = ratings[ratings['movieID'].isin(neighbourds[:,1])]
    nearestMovies = nearestMovies[nearestMovies.userID == user]
    if nearestMovies.empty:
        print('vacio')
        return 0.0
    return nearestMovies.rating.mean()

def getNearestNeighbords(matrix, element):

    bestNeigbours = np.zeros((NUMBERNEIGHBOURDS,2))

    minSim = 0
    posMin = 0

    for i in range(matrix.shape[0]):
        candidate = matrix.iloc[i]

        if candidate.name == element.name:
            continue

        sim = returnSim(element, candidate)
        if(sim > minSim):
            bestNeigbours[posMin][0] = sim
            bestNeigbours[posMin][1] = candidate.name

            posMin = np.unravel_index(bestNeigbours.argmin(), bestNeigbours.shape)[0]
            minSim = bestNeigbours[posMin][0]
    return bestNeigbours

def returnSim(one, two):
    # eliminate the last one that have the center value
    sim = []
    for i in range(one.shape[0]-1):
        sim.append(np.dot(one[i], two[i]))
    return np.mean(sim)

returnModel()