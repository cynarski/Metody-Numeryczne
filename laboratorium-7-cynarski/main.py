import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów

    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    try:
        if x.shape != y.shape or type(x) != np.ndarray or type(x) != type(y):
            return None
        a = np.ndarray([len(x) - 1, ])
        b = np.ndarray([len(x) - 1, ])
        for i in range(len(x) - 1):
            a[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            b[i] = y[i] - a[i]*x[i]
        return a, b
    except:
        return None


def cubic_spline(x: np.ndarray, y: np.ndarray, tol=1e-100):
    """
    Interpolacja splajnów cubicznych

    Returns:
    b współczynnik przy x stopnia 1
    c współczynnik przy x stopnia 2
    d współczynnik przy x stopnia 3
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape == y.shape:
            x = np.array(x)
            y = np.array(y)

            if np.any(np.diff(x) < 0):
                idx = np.argsort(x)
                x = x[idx]
                y = y[idx]

            size = len(x)
            delta_x = np.diff(x)
            delta_y = np.diff(y)

            A = np.zeros(shape=(size, size))
            b = np.zeros(shape=(size, 1))
            A[0, 0] = 1
            A[-1, -1] = 1

            for i in range(1, size - 1):
                A[i, i - 1] = delta_x[i - 1]
                A[i, i + 1] = delta_x[i]
                A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])
                b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])

            c = jacobi(A, b, np.zeros(len(A)), tol=tol, n_iterations=1000)

            d = np.zeros(shape=(size - 1, 1))
            b = np.zeros(shape=(size - 1, 1))
            for i in range(0, len(d)):
                d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
                b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

            return b.squeeze(), c.squeeze(), d.squeeze()

        else:
            return None
    else:
        return None


def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Iteracyjne rozwiązanie równania Ax=b dla zadanego x0

    Returns:
    x - estymowane rozwiązanie
    """

    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol + 1

    while (x_diff > tol) and (counter < n_iterations):
        for i in range(0, n):
            s = 0
            for j in range(0, n):
                if i != j:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]

        counter += 1
        x_diff = (np.sum((x - x_prev) ** 2)) ** 0.5
        x_prev = x.copy()

    return x





def chebyshev_nodes(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)

    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.

    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0:

        vector = np.ndarray(n + 1, )

        for k in range(n + 1):
            vector[k] = np.cos(k * np.pi / n)

        return vector

    else:
        return None


def bar_czeb_weights(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)

    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.

    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0:
        vector = np.ones(n + 1, )
        vector[0] = 1 / 2
        vector[-1] = 1 / 2
        w = np.ndarray(n + 1, )
        for j in range(n + 1):
            w[j] = ((-1) ** j) * vector[j]

        return w

    else:

        return None


def barycentric_inte(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n.

    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0

    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if all(isinstance(i, np.ndarray) for i in [xi, yi, wi, x]):

        if xi.shape == yi.shape == wi.shape:
            Y = []
            for x in np.nditer(x):
                L = wi / (x - xi)

                Y.append(yi @ L / sum(L))

            Y = np.array(Y)
            return Y

    else:

        return None


def L_inf(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray]) -> float:
    """Obliczenie normy  L nieskończonośćg.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.

    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)

    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):

        return abs(xr - x)

    elif isinstance(xr, list) and isinstance(x, list):

        return abs(max(xr) - max(x))

    elif isinstance(xr, np.ndarray) and isinstance(x, np.ndarray):

        if xr.shape == x.shape:

            return max(abs(xr - x))
        else:

            return np.NaN

    else:

        return np.NaN