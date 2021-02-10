# -*- coding: utf-8 -*-
import pandas as pd
import sys


def average(filename, column):
    """
    Retourne la moyenne de la colonne "column" à partir du
    fichier csv "filename"

    Parameters
    ----------
    filename : str
        CSV filename.
    column : str
        Column on which we process the mean.

    Returns
    -------
    float mean value.

    """
    df = pd.read_csv(filename)
    return df[column].mean()


if __name__ == "__main__":
    filename = sys.argv[1]
    column = sys.argv[2]
    print(f"On récupère la moyenne de la colonne {column}" +
          f" dans le fichier {filename}")
    
    print(average(filename, column))