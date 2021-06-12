"""This module contains classes for functions which the MLP can learn."""

import numpy


class LinearA:
    """Class for the linear function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = x-2`"""
    name = 'LinearA'

    @staticmethod
    def f(x):
        return -x + 2


class QuadraticA:
    """Class for the quadratic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = (x-3)^2 + 1`"""
    name = 'QuadraticA'

    @staticmethod
    def f(x):
        return (x-3)*(x-3) + 1


class QuadraticB:
    """Class for the quadratic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = (x-1)^2 \cdot 0.3 - 1`"""
    name = 'QuadraticB'

    @staticmethod
    def f(x):
        return (x-1)*(x-1)*0.3-1


class LinearA2D:
    """Class for the linear function :math:`f: \in \mathbb{R}^2 \\rightarrow \mathbb{R}^2, f(x) = \\begin{bmatrix} -0.1 & -1.0 \\\ 1.0 & -0.1 \end{bmatrix} \cdot x`"""
    name = 'Linear2DA'

    @staticmethod
    def f(x):
        return numpy.matmul([[-0.1, -1.0], [1.0, -0.1]], x)
