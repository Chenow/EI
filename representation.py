from models import pca_reduction
import matplotlib.pyplot as plt
from sklearn import preprocessing

def tracer_figure_20_dernier_jours(model, model_name, X, y):

    X_time = X['Time']
    X_pca = pca_reduction(X)
    
    last_20_days = X_pca[-48*20:]
    y_predicted =  model.predict(last_20_days)
    plt.cla()
    plt.scatter(X_time[-48*20:], y[-48*20:], c="blue", label="vérification")
    plt.plot(X_time[-48*20:], y_predicted, c="red", label="prédiction")
    plt.legend()
    plt.gcf().set_size_inches(25, 16)

    plt.savefig('test_graphe_' + model_name + '.png')


