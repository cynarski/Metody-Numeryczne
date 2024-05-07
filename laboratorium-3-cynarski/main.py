import numpy as np
import scipy
import pickle
import math
from typing import Union, List, Tuple


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
  
    if isinstance(n,int) and isinstance(c,(int,float)):

        b = 2 ** n
        P1 = b - b + c
        P2 = b + c - b

        return np.abs(P1 - P2)

    else:
        return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """


    if isinstance(x,(int,float)) and isinstance(n,int):

        if n < 0:
            return np.NaN

        result = 0
        for i in range(n):

            result += (1/math.factorial(i)) * x ** i

        return result
    else:
        return np.NaN


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(k,int) and isinstance(x,(int,float)):

        if k < 0:
            return np.NaN

        elif k == 0:
            return 1

        elif k == 1:
            return np.cos(x)

        else:
            return 2 * np.cos(x) * coskx1(k - 1,x) - coskx1(k - 2,x)

    else:
        return np.NaN


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(k, int) and isinstance(x, (int, float)):

        if k < 0:
            return np.NaN

        elif k == 0:
            return 1,0

        elif k == 1:
            return np.cos(x), np.sin(x)

        else:
            return np.cos(x) * coskx2(k - 1,x)[0] - np.sin(x) * coskx2(k - 1,x)[1] , np.sin(x) * coskx2(k - 1,x)[0] + np.cos(x) * coskx2(k - 1,x)[1]

    else:
        return np.NaN

