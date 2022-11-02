import pandas as pd

def main():
    #0.90
    dataClousterViewers = pd.read_pickle('data_clouster_predictions.pkl')
    #0.92
    dataClousterMovies = pd.read_pickle('data_contente_based_predictions.pkl')

    data = pd.DataFrame()
    for i in range(dataClousterViewers.shape[0]):
        if dataClousterViewers.iloc[i].values[0] == 0:
            data = data.append({'values' : dataClousterMovies.iloc[i].values[0]}, ignore_index=True)
        elif dataClousterMovies.iloc[i].values[0] == 0:
            data = data.append({'values' : dataClousterViewers.iloc[i]}, ignore_index= True)
        else:
            d = (dataClousterViewers.iloc[i].values[0] + dataClousterMovies.iloc[i].values[0]) / 2
            data = data.append({'values' : round(d,1)}, ignore_index = True)

    #0.85
    writeData(data, 'Resources/MixClouster.dat')

    print('nais')

    return


def writeData(data, nameFile):
    file = open(nameFile,'w')
    for i in range(data.shape[0]):
        file.write(str(float(data.iloc[i].values[0])) + '\n')
    file.close()


main()