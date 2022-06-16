from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_reduction(X):
    """ Reduces the number of parameters used to train our models.

    Args:
        X (pandas.Dataframe): contains all the column datas but "mean" from the datas file.
    Returns:
        pandas.DataFrame: contains fewer columns than X but keeps 95% of the data information
    """    

    pca=PCA(n_components=7)
    X_pca = pca.fit_transform(preprocessing.scale(X))

    return X_pca


def train_and_test_model(X, y, model):
    """ train and test a model.

    Args:
        X (panda.Dataframe): contains all the column datas  but "mean" from the datas file.
        y (panda.Dataframe): contain the "mean column from the training file.
        model (sklearn.Model): model to be trained 

    Returns:
        score: returns the mean_squared_error of 'model'
    """    

    X_pca = pca_reduction(X)

    # sératation du dataset en train et test
    limit = int(len(y)*0.8)
    X_train, X_test = preprocessing.scale(X_pca[:limit]), preprocessing.scale(X_pca[limit:])
    y_train, y_test = y[:limit], y[limit:]

    model.fit(X_train, y_train)

    return mean_squared_error(y_test,model.predict(X_test))


def evaluate_all_models(X, y, models):
    """train and test all models in 'model'.

    Args:
       X (panda.Dataframe): contains all the column datas  but "mean" from the datas file.
        y (panda.Dataframe): contain the "mean column from the training file.
        models (list of sklearn.Models): list of models to be trained 

    Returns:
        score: returns the list of mean_squared_errors of the models in 'models'
    """    
    mean_squared_errors = {}

    for model_name in models:
        mean_squared_errors[model_name] = train_and_test_model(X, y, models[model_name])

    return mean_squared_errors
