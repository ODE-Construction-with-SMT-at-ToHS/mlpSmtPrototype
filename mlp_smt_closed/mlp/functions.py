"""This module contains classes for functions which the MLP can learn."""

import numpy
from abc import ABC, abstractmethod

import numpy as np


class FuncClass(ABC):
    """Abstract parent class for all functions"""
    @property
    @abstractmethod
    def name(self):
        """This function returns the name of the function."""
        pass

    @property
    @abstractmethod
    def dimension(self):
        """This function returns the dimensionality of the function. Note that at this point only a very small portion
        of multidimensional functions are supported, namely all of the form :math:`f: x \in \mathbb{R}^n \mapsto a\cdot
        x+b` with :math:`a \in \mathbb{R}^{n \\times n}, b \in \mathbb{R}^n`
        """
        pass

    @property
    @abstractmethod
    def degree(self):
        """This function returns the degree of the function"""
        pass

    @abstractmethod
    def f(self, x):
        """This function returns the value of the :math:`f` for input :math:`x`"""
        pass


# Linear 1D Functions
class LinearA(FuncClass):
    """Class for the linear function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = -x+2`"""
    def name(self):
        """returns 'LinearA'"""
        return 'LinearA'

    def dimension(self):
        """returns 1"""
        return 1

    def degree(self):
        """returns 1"""
        return 1

    def f(self, x):
        return -x + 2


# quadratic 1D functions
class QuadraticA(FuncClass):
    """Class for the quadratic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = 0.1 \cdot (x^2-6x+10)`
    """

    def name(self):
        """returns 'QuadraticA'"""
        return 'QuadraticA'

    def dimension(self):
        """returns 1"""
        return 1

    def degree(self):
        """returns 2"""
        return 2

    def f(self, x):
        return 0.1 * (x**2 - 6*x + 10)


class PolyDeg3(FuncClass):
    """Class for the cubic function :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R},
    0.025\cdot(0.4\cdot(x^3) - 2\cdot(x^2) + 3\cdot x - 5)`
    """
    def name(self):
        """returns 'PolyDeg3'"""
        return 'PolyDeg3'

    def dimension(self):
        """returns 1"""
        return 1

    def degree(self):
        """returns 3"""
        return 3

    def f(self, x):
        return 0.025*(0.4*(x**3) - 2*(x**2) + 3*x - 5)


# Linear 2D Functions
class LinearA2D(FuncClass):
    """Class for the linear function :math:`f: \in \mathbb{R}^2 \\rightarrow \mathbb{R}^2,
    f(x) = \\begin{bmatrix} -0.1 & -1.0 \\\ 1.0 & -0.1 \end{bmatrix} \cdot x`
    """

    def name(self):
        """returns 'LinearA2D'"""
        return 'LinearA2D'

    def dimension(self):
        """returns 2"""
        return 2

    def degree(self):
        """returns 1"""
        return 1

    def f(self, x):
        return numpy.matmul([[-0.1, -1.0], [1.0, -0.1]], x)


class Platoon(FuncClass):
    """This function is taken is a `benchmark from the HyPro project <https://ths.rwth-aachen.de/research/projects/hypro/n_vehicle_platoon/>`_ """
    def name(self):
        """returns 'Platoon'"""
        return 'Platoon'

    def dimension(self):
        """returns 15"""
        return 15

    def degree(self):
        """returns 1"""
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


class PolyGen(FuncClass):
    """Generic class for Polynomial Functions. This class is not for Network training purposes but only necessary
    because of bad construction. Only dimension and degree should be used.
    """

    def __init__(self, degree):
        self.var_degree = degree

    def name(self):
        """returns error, should not be used"""
        return 'Error, PolyGen.name() called, this should not happen!'

    def dimension(self):
        """returns 1, as only 1D polynomials are supported"""
        return 1

    def degree(self):
        """returns degree passed at construction time"""
        return self.var_degree

    def f(self, x):
        """returns error, should not be used"""
        return 'Error, PolyGen.f(x) called, this should not happen'


class LinGen(FuncClass):
    """Generic class for Linear Functions. This class is not for Network training purposes but only necessary
    because of bad construction. Only dimension and degree should be used.
    """

    def __init__(self, dimension):
        self.var_dimension = dimension

    def name(self):
        """returns error, should not be used"""
        return 'Error, LinGen.name() called, this should not happen!'

    def dimension(self):
        """returns the dimension passed at construction time"""
        return self.var_dimension

    def degree(self):
        """return 1 as this is a linear function"""
        return 1

    def f(self, x):
        """returns error, should not be used"""
        return 'Error, LinGen.f(x) called, this should not happen'
