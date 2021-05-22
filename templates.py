from abc import ABC, abstractmethod
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


class LinearTemplate(Template):

    # f(x) = a*x + b
    def __init__(self):
        self.params = {'a': 0, 'b': 0}
        self.input_vars = [Real('x')]
        self.output_vars = [Real('y')]
        self.param_vars = {'a': Real('a'), 'b': Real('b')}
    
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
