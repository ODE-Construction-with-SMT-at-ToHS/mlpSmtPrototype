"""This module contains classes for functions which the MLP can learn."""

import numpy
from abc import ABC, abstractmethod

import numpy as np


class FuncClass(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def degree(self):
        pass

    @abstractmethod
    def f(self, x):
        pass


# Linear 1D Functions
class LinearA(FuncClass):
    """Class for the linear function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = x-2`"""
    def name(self):
        return 'LinearA'

    def dimension(self):
        return 1

    def degree(self):
        return 1

    def f(self, x):
        return -x + 2


class LinearB(FuncClass):
    def name(self):
        return 'LinearB'

    def dimension(self):
        return 1

    def degree(self):
        return 1

    def f(self, x):
        return -3 * x + 8


# quadratic 1D functions
class QuadraticA(FuncClass):
    """Class for the quadratic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = (x-3)^2 + 1`"""

    def name(self):
        return 'QuadraticA'

    def dimension(self):
        return 1

    def degree(self):
        return 2

    def f(self, x):
        return 0.1 * (x**2 - 6*x + 10)


class QuadraticB(FuncClass):
    """Class for the quadratic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = (x-1)^2 \cdot 0.3 - 1`"""
    def name(self):
        return 'QuadraticB'

    def dimension(self):
        return 1

    def degree(self):
        return 2

    def f(self, x):
        return (x-1)*(x-1)*0.3-1

class PolyDeg3(FuncClass):
    def name(self):
        return 'PolyDeg3'

    def dimension(self):
        return 1

    def degree(self):
        return 3

    def f(self, x):
        return 0.025*(0.4*(x**3) - 2*(x**2) + 3*x - 5)


# Linear 2D Functions
class LinearA2D(FuncClass):
    """Class for the linear function :math:`f: \in \mathbb{R}^2 \\rightarrow \mathbb{R}^2, f(x) = \\begin{bmatrix} -0.1 & -1.0 \\\ 1.0 & -0.1 \end{bmatrix} \cdot x`"""

    def name(self):
        return 'LinearA2D'

    def dimension(self):
        return 2

    def degree(self):
        return 1

    def f(self, x):
        return numpy.matmul([[-0.1, -1.0], [1.0, -0.1]], x)


class LinearB2D(FuncClass):
    def name(self):
        return 'LinearB2D'

    def dimension(self):
        return 2

    def degree(self):
        return 1

    def f(self, x):
        return numpy.matmul([[-0.1, -1.0], [1.0, -0.1]], x)


class Platoon(FuncClass):
    def name(self):
        return 'Platoon'

    def dimension(self):
        return 15

    def degree(self):
        return 1

    def f(self, x):

        A = np.array([[0,             1,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0],
                      [0,             0,              -1,             0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0],
                      [1.7152555329,  3.9705119979,   -4.3600526739,  -0.9999330812,  -1.5731541104,  0.2669165553,   -0.2215507198,  -0.4303855023,  0.0669078193,   -0.0881500219,  -0.1881468451,  0.0322187056,   -0.0343095071,  -0.0767587194,  0.0226660281],
                      [0,             0,              0,              0,              1,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0],
                      [0,             0,              1,              0,              0,              -1,             0,              0,              0,              0,              0,              0,              0,              0,              0],
                      [0.7153224517,  2.3973578876,   0.2669165553,   1.4937048131,   3.5401264957,   -4.2931448546,  -1.0880831031,  -1.7613009555,  0.2991352608,   -0.2558602268,  -0.5071442217,  0.0895738474,   -0.0881500219,  -0.1881468451,  0.0548847337],
                      [0,             0,              0,              0,              0,              0,              0,              1,              0,              0,              0,              0,              0,              0,              0],
                      [0,             0,              0,              0,              0,              1,              0,              0,              -1,             0,              0,              0,              0,              0,              0],
                      [0.493771732,   1.9669723853,   0.0669078193,   0.6271724298,   2.2092110425,   0.2991352608,   1.4593953061,   3.4633677762,   -4.2704788265,  -1.0880831031,  -1.7613009555,  0.3218012889,  -0.2215507198,   -0.4303855023,  0.121792553],
                      [0,             0,              0,              0,              0,              0,              0,              0,              0,              0,              1,              0,              0,              0,              0],
                      [0,             0,              0,              0,              0,              0,              0,              0,              1,              0,              0,              -1,             0,              0,              0],
                      [0.40562171,    1.7788255402,   0.0322187056,   0.4594622249,   1.8902136659,   0.0895738474,   0.6271724298,   2.2092110425,   0.3218012889,   1.4937048131,   3.5401264957,   -4.2382601209,  -0.9999330812,  -1.5731541104,  0.3887091083],
                      [0,             0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              1,              0],
                      [0,             0,              0,              0,              0,              0,              0,              0,              0,              0,              0,              1,              0,              0,              -1],
                      [0.371312203,   1.7020668208,   0.0226660281,   0.40562171,     1.7788255402,   0.0548847337,   0.493771732,    1.9669723853,   0.121792553,    0.7153224517,   2.3973578876,   0.3887091083,   1.7152555329,   3.9705119979,   -3.9713435656]
                     ])
        return numpy.matmul(A, x)



class Brusselator:
    """ Class representing the dynamics of the Brusselator"""
    name = 'Brusselator'

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
    
    def f(self, x):
        x_prime = self.a + x[0]*x[0]*x[1] - self.b*x[0] - x[0]
        y_prime = self.b*x[0] - x[0]*x[0]*x[1]
        return (x_prime, y_prime)
