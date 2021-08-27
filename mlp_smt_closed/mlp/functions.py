"""This module contains classes for functions which the MLP can learn."""

import numpy
from abc import ABC, abstractmethod


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
        return x**2 - 6*x + 10


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


# Linear 2D Functions
class LinearA2D(FuncClass):
    """Class for the linear function :math:`f: \in \mathbb{R}^2 \\rightarrow \mathbb{R}^2, f(x) = \\begin{bmatrix} -0.1 & -1.0 \\\ 1.0 & -0.1 \end{bmatrix} \cdot x`"""

    def name(self):
        return 'Linear2DA'

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
