import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from transformers import gestion_cyclique

def get_raw_datas(path, new=False, temps_cyclique =False):   
    """fonction qui récupère les données X et y à partir du path d'un fichier CSV

    Args:
        path (String): le chemin d'acces du CSV
        new (bool, optional): si on prends le dataset de base ou le dataset modifié. Defaults to False.
        temps_cyclique (bool, optional): si on interprète les données temporelles avec des splines. Defaults to False.

    Returns:
        pandas.dataframe: les dataframes pandas X les entrées d'entrainement et y les veleurs mean
    """

    if new:
        df = pd.read_csv(path + '_modifie.csv', sep=";", decimal=',')
    else:
        df = pd.read_csv(path + '.csv', sep=",")
    
    
    if temps_cyclique:
        df = gestion_cyclique(df,"jour")
        df = gestion_cyclique(df,"semaine")
        df = gestion_cyclique(df,"annee")

    y = df['mean']
    df = df.drop(columns=['Date'])

    X = df.loc[: , df.columns != 'mean']
    return X, y