from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd


def periodic_spline_transformer(period, n_splines=None, degree=3):
    """transforme une fonction periodique en splines
    cette fonction viens de cette page : https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html

    Args:
        period (int): la periode du phénomène observé
        n_splines (int, optional): nombre de splines en sortie. Defaults to None.
        degree (int, optional):degrés. Defaults to 3.

    Returns:
        _type_: _description_
    """    
    # Par défault on pose le nombre de spline à la periode
    if n_splines is None:
        n_splines = period

    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )




def gestion_cyclique(df, cycle):
    """ajoute des splines à un dataframe pour prendre en compte les cycles

    Args:
        df (pandas.DataFrame): dataframe où ajouter des colonne
        cycle (String): cycle à prendre en compte

    Returns:
        pandas.DataFrame: dataframe modifié
    """     

    #on met à jour les paramètres en fonction du cycle
    if cycle == "annee":
        colonne = "Posan"
        periode = 365
        n_splines=48
    elif cycle == "semaine":
        colonne = "Posan"
        periode = 7
        n_splines=7
    elif cycle == "jour":
        colonne = "Time"
        periode = 48
        n_splines=48
    else:
        raise ValueError("cycle n'est pas une valeur attendu")
    
    # On récupère un DataFrame cycl_df qui est la colonne associée à notre cycle
    cycl_df = df.loc[:,[colonne]]
    #print(cycl_df)
    
    # On créé les colonnes liées aux splines
    splines = periodic_spline_transformer(periode, n_splines=n_splines).fit_transform(cycl_df)
    splines_df = pd.DataFrame(
        splines,
        columns=[cycle + f"Spline_{i}" for i in range(splines.shape[1])],
    )
    #print(splines_df)

    # On concatène nos splines au DataFrame de départ
    df =pd.concat([splines_df,df], axis="columns")
    return df