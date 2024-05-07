import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m, m))
        b = np.random.randint(0, 9, (m,))
        max_in_rows_and_cols = np.sum(A, axis=0) + np.sum(A, axis=1)
        A = A + np.diag(max_in_rows_and_cols)
        return A, b
    else:
        return None


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray):
        if len(A.shape) == 2:
            if A.shape[0] == A.shape[1]:
                return all((2 * np.abs(np.diag(A))) >= sum(np.abs(A), 1))
            else:
                return None
        else:
            return None
    else:
        return None


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 9, (m, m))
        b = np.random.randint(0, 9, (m,))
        A_symmetric = A + A.T
        return A_symmetric, b
    else:
        return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray):
        if len(A.shape) == 2:
            if A.shape[0] == A.shape[1]:
                return np.allclose(A, A.T, 1e-05, 1e-05)
            else:
                return None
        else:
            return None
    else:
        return None

def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if all(isinstance(i, np.ndarray) for i in [A, b, x_init]) and isinstance(epsilon, float) and isinstance(maxiter,
                                                                                                            int):
        if maxiter > 0 and epsilon > 0 and A.shape[0] == A.shape[1] and A.shape[1] == b.shape[0] and b.shape[0] == \
                x_init.shape[0]:
            D = np.diag(np.diag(A))
            LU = A - D
            x = x_init
            D_inv = np.diag(1 / np.diag(D))

            for i in range(maxiter):
                x_new = np.dot(D_inv, b - np.dot(LU, x))

                if np.linalg.norm(x_new - x) < epsilon:
                    return x_new, i
                x = x_new
            return x, maxiter
        else:
            return None
    else:
        return None

def random_matrix_Ab(m:int):
    if isinstance(m, int) and m > 0:
        return np.random.randint(500, size=(m, m)), np.random.randint(500, size=(m,))
    else:
        return None



