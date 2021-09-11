"""
This module contains ``z3``-encodings of the templates for the closed form of the MLP. The function
``optimize_template`` from ``logic.py`` tries to fit the parameters of a given templates to the input/output relation of
the MLP.
"""

from abc import ABC, abstractmethod
from os import name
from z3 import *


# abstract class for all templates
class Template(ABC):

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def input_variables(self):
        pass

    @abstractmethod
    def output_variables(self):
        pass

    @abstractmethod
    def param_variables(self):
        pass

    @abstractmethod
    def func(self, x):
        pass

    @abstractmethod
    def smt_encoding(self):
        pass

class PolynomialTemplate(Template):
    """
    Class for PolynomialTemplates
    """

    def __init__(self, encoding='Real', degree=1, variables=1) -> None:
        self.encoding = encoding
        self.degree = degree
        self.variables = variables

        input_var_names = ['x_{}'.format(i) for i in range(variables)]
        output_var_names = ['y']
        param_var_names = ['a']
        # For each variable we iteratively add each possible exponent
        for i in range(variables):
            extended_vars = []
            for j in range(degree+1):
                extended_vars.extend([a + ',' + str(j) for a in param_var_names])
            param_var_names = extended_vars
        
        print(param_var_names)
        self.param_var_names = param_var_names
        self.params = {name: 1 for name in self.param_var_names}

        if encoding == 'FP':
            self.input_vars = [FP(name, Float32()) for name in input_var_names]
            self.output_vars = [FP(name, Float32()) for name in output_var_names]
            self.param_vars = {name: FP(name, Float32()) for name in param_var_names}
        elif encoding == 'Real':
            self.input_vars = [Real(name) for name in input_var_names]
            self.output_vars = [Real(name) for name in output_var_names]
            self.param_vars = {name: Real(name) for name in param_var_names}
    
    # overriding abstract method
    def func(self, x):
        y = 0
        ctr = 0
        var_indeces = [i for i in range(self.variables)]
        var_indeces.sort(reverse=True)
        for name in self.param_vars:
            a = self.params[name]
            ctr_cpy = ctr
            # The index of a parameter can be mapped to the corresponding exponents
            for index in var_indeces:
                exponent = int(ctr_cpy/((self.degree+1)**index))
                ctr_cpy = ctr_cpy - exponent*((self.degree+1)**index)
                a = a*(x[index]**exponent)
            ctr += 1
            y += a

        return y

    # overriding abstract method
    def smt_encoding(self):
        y = 0
        ctr = 0
        var_indeces = [i for i in range(self.variables)]
        var_indeces.sort(reverse=True)
        for name in self.param_vars:
            a = self.params[name]
            ctr_cpy = ctr
            # The index of a parameter can be mapped to the corresponding exponents
            for index in var_indeces:
                exponent = int(ctr_cpy/((self.degree+1)**index))
                ctr_cpy = ctr_cpy - exponent*((self.degree+1)**index)
                if not exponent == 0:
                    a = a*(self.input_vars[index]**exponent)
            ctr += 1
            y += a

        encoding = self.output_vars[0] == y
        return encoding

    def generic_smt_encoding(self, input_value, output_var):
        y = 0
        ctr = 0
        var_indeces = [i for i in range(self.variables)]
        var_indeces.sort(reverse=True)
        for name in self.param_vars:
            a = self.param_vars[name]
            ctr_cpy = ctr
            # The index of a parameter can be mapped to the corresponding exponents
            for index in var_indeces:
                exponent = int(ctr_cpy/((self.degree+1)**index))
                ctr_cpy = ctr_cpy - exponent*((self.degree+1)**index)
                if not exponent == 0:
                    a = a*(input_value[index]**exponent)
            ctr += 1
            y += a

        encoding = output_var[0] == y
        return encoding

    # overriding abstract method
    def get_params(self):
        return self.params

    # overriding abstract method
    def set_params(self, params):
        print(self.params)
        self.params = params

    # overriding abstract method
    def input_variables(self):
        return self.input_vars

    # overriding abstract method
    def output_variables(self):
        return self.output_vars

    # overriding abstract method
    def param_variables(self):
        return self.param_vars
    
    def test_encoding(self, input):
        formula = [self.smt_encoding()]
        for i in range(len(self.input_vars)):
            formula.append(input[i] == self.input_vars[i])

        solver = Solver()
        for e in formula:
            solver.add(e)
        
        res = solver.check()
        print(formula)
        if res != sat:
            print('ERROR. Template-Encoding is not satisfiable.')
        else:
            fo_model = solver.model()
            print(fo_model.eval(self.output_vars[0], model_completion=True))


class LinearTemplate(Template):
    """
    Class representing the set of linear functions :math:`f` with :math:`f: \in \mathbb{R} \\rightarrow \mathbb{R}, f(x) = a \cdot x + b, a,b \in \mathbb{R}`
    """

    # f(x) = a*x + b
    def __init__(self, encoding='Real'):
        self.encoding = encoding

        self.params = {'a': 1, 'b': 1}

        input_var_names = ['x']
        output_var_names = ['y']
        param_var_names = ['a','b']

        if encoding == 'FP':
            self.input_vars = [FP(name, Float32()) for name in input_var_names]
            self.output_vars = [FP(name, Float32()) for name in output_var_names]
            self.param_vars = {name: FP(name, Float32()) for name in param_var_names}
        elif encoding == 'Real':
            self.input_vars = [Real(name) for name in input_var_names]
            self.output_vars = [Real(name) for name in output_var_names]
            self.param_vars = {name: Real(name) for name in param_var_names}
    
    # overriding abstract method
    def func(self, x):
        y = self.params['a']*x[0] + self.params['b']
        return y

    # overriding abstract method
    def smt_encoding(self):
        encoding = self.output_vars[0] == self.params['a']*self.input_vars[0] + self.params['b']
        return encoding

    def generic_smt_encoding(self, input_value, output_var):
        encoding = output_var[0] == self.param_vars['a'] * input_value[0] + self.param_vars['b']
        return encoding

    # overriding abstract method
    def get_params(self):
        return self.params

    # overriding abstract method
    def set_params(self, params):
        self.params = params

    # overriding abstract method
    def input_variables(self):
        return self.input_vars

    # overriding abstract method
    def output_variables(self):
        return self.output_vars

    # overriding abstract method
    def param_variables(self):
        return self.param_vars

class Linear2DTemplate(Template):

    # f(x) = A*x + b
    def __init__(self, encoding='Real'):

        input_var_names = ['x1','x2']
        output_var_names = ['y1','y2']
        param_var_names = ['a11','a12','a21','a22','b1','b2']

        self.params = {name: 0 for name in param_var_names}

        if encoding == 'FP':
            self.input_vars = [FP(name, Float32()) for name in input_var_names]
            self.output_vars = [FP(name, Float32()) for name in output_var_names]
            self.param_vars = {name: FP(name, Float32()) for name in param_var_names}
        elif encoding == 'Real':
            self.input_vars = [Real(name) for name in input_var_names]
            self.output_vars = [Real(name) for name in output_var_names]
            self.param_vars = {name: Real(name) for name in param_var_names}
    
    # overriding abstract method
    def func(self, x):
        y1 = self.params['a11']*x[0] + self.params['a12']*x[1]  + self.params['b1']
        y2 = self.params['a21']*x[0] + self.params['a22']*x[1]  + self.params['b2']
        return y1, y2

    # overriding abstract method
    def smt_encoding(self):
        y1 = self.output_vars[0] == self.params['a11']*self.input_vars[0] + self.params['a12']*self.input_vars[1]  + self.params['b1']
        y2 = self.output_vars[1] == self.params['a21']*self.input_vars[0] + self.params['a22']*self.input_vars[1]  + self.params['b2']
        return And(y1,y2)

    def generic_smt_encoding(self, input_value, output_var):
        y1 = output_var[0] == self.param_vars['a11']*input_value[0] + self.param_vars['a12']*input_value[1]  + self.param_vars['b1']
        y2 = output_var[1] == self.param_vars['a21']*input_value[0] + self.param_vars['a22']*input_value[1]  + self.param_vars['b2']
        return And(y1,y2)

    # overriding abstract method
    def get_params(self):
        return self.params

    # overriding abstract method
    def set_params(self, params):
        self.params = params

    # overriding abstract method
    def input_variables(self):
        return self.input_vars

    # overriding abstract method
    def output_variables(self):
        return self.output_vars

    # overriding abstract method
    def param_variables(self):
        return self.param_vars
