import numpy as np
import pandas as pd

def getData(read = True):
    ratingsTrain = pd.read_csv("Resources/additional_files/train.dat", sep="\s+", header=0, usecols=["userID","movieID", "rating"])
    ratingsTest = pd.read_csv("Resources/additional_files/test.dat", sep="\s+", header=0, usecols=["userID","movieID"])
        
    if not read:
        movie_actor = pd.read_csv("Resources/additional_files/movie_actors.dat",encoding='cp1252', sep="\t", header=0, usecols=["movieID","actorID","ranking"])
        movie_directors = pd.read_csv("Resources/additional_files/movie_directors.dat",encoding='cp1252', sep="\t", header=0, usecols=["movieID","directorID"])
        movie_genres = pd.read_csv("Resources/additional_files/movie_genres.dat", sep="\t", header=0, usecols=["movieID","genre"])
        movie_tags = pd.read_csv("Resources/additional_files/movie_tags.dat", sep="\t", header=0, usecols=["movieID","tagID", "tagWeight"])
        #tags = pd.read_csv("Resources/additional_files/tags.dat", encoding="cp1252", sep="\t", header=0, usecols=["id","value"])

        matrix = pd.DataFrame()
        matrix['movieID'] = ratingsTrain.movieID.unique()
        matrix = matrix.set_index('movieID')
        matrix['actor'] = None
        actor_dic = movie_actor.actorID.unique().tolist()
        matrix['director'] = None
        director_dic = movie_directors.directorID.unique().tolist()
        matrix['genre'] = None
        genre_dic = movie_genres.genre.unique().tolist()
        matrix['tags'] = None
        tag_dic = movie_tags.tagID.unique().tolist()
        for i in range(matrix.shape[0]):
            movie = matrix.iloc[i]
            matrix.at[movie.name,'actor'] = returnElementsOnDiccionary(actor_dic, movie_actor[movie_actor.movieID == movie.name].actorID.values)
            matrix.at[movie.name,'director'] = returnElementsOnDiccionary(director_dic, movie_directors[movie_directors.movieID == movie.name].directorID.values)
            matrix.at[movie.name,'genre'] = returnElementsOnDiccionary(genre_dic, movie_genres[movie_genres.movieID == movie.name].genre.values)
            matrix.at[movie.name,'tags'] = returnElementsOnDiccionary(tag_dic, movie_tags[movie_tags.movieID == movie.name].tagID.values)
            
        matrix.to_pickle('data_contente_based.pkl')
    else:
        matrix = pd.read_pickle('data_contente_based.pkl')
    return ratingsTrain, ratingsTest, matrix

def returnElementsOnDiccionary(diccionary, movie):
    wordsPresents = np.zeros(len(diccionary))
    for i in range(len(movie)):
        index = diccionary.index(movie[i])
        wordsPresents[index] = 1.0/len(movie)
    return wordsPresents

def getTrainingRatingMatrix():
    return pd.read_csv("Resources/additional_files/train.dat", sep="\s+", header = 0, usecols=["movieID","rating", "userID"])

def getTrainingRatingMatrixAsTable():
    data = getTrainingRatingMatrix()

    data = data.pivot_table(columns='movieID', index='userID', values = 'rating', fill_value=0)
    return data

def getTestMatrix():
    return pd.read_csv("Resources/additional_files/test.dat", sep="\s+", header = 0, usecols=["movieID", "userID"])

#getData()