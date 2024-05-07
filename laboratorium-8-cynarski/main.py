import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple

'''
Do celów testowych dla elementów losowych uzywaj seed = 24122022
'''

def random_matrix_by_egval(egval_vec: np.ndarray):
    """Funkcja z pierwszego zadania domowego
    Parameters:
    egval_vec : wetkor wartości własnych
    Results:
    np.ndarray: losowa macierza o zadanych wartościach własnych 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(egval_vec, (np.ndarray, List)):

        for x in egval_vec:
            if isinstance(x, str) == True:
                return None

        np.random.seed(24122022)
        J = np.diag(egval_vec)
        P = np.random.rand(len(egval_vec), len(egval_vec))
        div_P = np.linalg.inv(P)

        return P @ J @ div_P

    return None


def frob_a(coef_vec: np.ndarray):
    """Funkcja z drugiego zadania domowego
    Parameters:
    coef_vec : wetkor wartości wspołczynników
    Results:
    np.ndarray: macierza Frobeniusa o zadanych wartościach współczynników wielomianu 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(coef_vec, np.ndarray):
        result = np.diag(np.ones(len(coef_vec) - 1), 1)
        result[-1, :] = -coef_vec[::-1]
        return result

    return None

    
def polly_from_egval(egval_vec: np.ndarray):
    """Funkcja z laboratorium 8
    Parameters:
    egval_vec: wetkor wartości własnych
    Results:
    np.ndarray: wektor współczynników wielomianu charakterystycznego
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(egval_vec,np.ndarray):
        return None

    result = np.poly(egval_vec)
    return result