from math import pi
import numpy as np


def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:

    Returns:
    float: pole powierzchni walca
    """
    if r > 0 and h > 0:
        return 2*pi*r**2 + 2*pi*r*h
    else:
        return np.NaN


def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    fibo = np.array([1, 1])
    if isinstance(n, int):
        if n <= 0:
            return None
        if n == 1:
            return fibo[0]
        if n == 2:
            return fibo
        if n > 2:
            fibo = [1.0, 1.0]
            for i in range(n - 2):
                fibo.append(fibo[-1] + fibo[-2])
        return np.array([fibo])

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej
    na podstawie parametru a.
    Szczegółowy opis w zadaniu 4.

    Parameters:
    a (float): wartość liczbowa

    Returns:
    touple: krotka zawierająca wyniki obliczeń
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a,1,-a],[0,1,1],[-a,a,1]])
    Mdet = np.linalg.det(M)
    Mt = np.transpose(M)
    if not Mdet:
        Minv =  np.NaN
    else:
        Minv = np.linalg.inv(M)
    result = (Minv,Mt,Mdet)
    return result
def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie
    z opisem zadania 7.

    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m < 0 or n < 0:
        return None

    if isinstance(m,int) and isinstance(n,int):
        Matrix = np.zeros(shape=(m,n))
        for i in range(m):
            for j in range(n):
                if i > j:
                    Matrix[i][j] = i
                else:
                    Matrix[i][j] = j


        return Matrix

    else:
        return None

