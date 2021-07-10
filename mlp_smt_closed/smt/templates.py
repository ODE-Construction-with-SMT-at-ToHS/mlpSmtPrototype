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
        self.input_vars = [FP('x', Float32())]
        self.output_vars = [FP('y', Float32())]
        self.param_vars = {'a': FP('a', Float32()), 'b': FP('b', Float32())}

        self.real_param_vars = {'a': Real('a'), 'b': Real('b')}
    
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

    def generic_real_smt_encoding(self, input_value, output_var):
        encoding = output_var[0] == self.real_param_vars['a'] * input_value[0] + self.real_param_vars['b']
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
    
    def real_param_variables(self):
        return self.real_param_vars

class Linear2DTemplate(Template):

    # f(x) = A*x + b
    def __init__(self):
        self.params = {'a11': 0,'a12': 0,'a21': 0,'a22': 0, 'b1': 0,'b2': 0}
        self.input_vars = [FP('x1', Float32()),FP('x2', Float32())]
        self.output_vars = [FP('y1', Float32()),FP('y2', Float32())]
        self.param_vars = {'a11': FP('a11', Float32()),'a12': FP('a12', Float32()),'a21': FP('a21', Float32()),
            'a22': FP('a22', Float32()), 'b1': FP('b1', Float32()),'b2': FP('b2', Float32())}

        self.real_param_vars = {'a11': Real('a11'), 'a12': Real('a12'), 'a21': Real('a21'),
            'a22': Real('a22'), 'b1': Real('b1'), 'b2': Real('b2')}
    
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

    def generic_real_smt_encoding(self, input_value, output_var):
        y1 = output_var[0] == self.real_param_vars['a11']*input_value[0] + self.real_param_vars['a12']*input_value[1]  + self.real_param_vars['b1']
        y2 = output_var[1] == self.real_param_vars['a21']*input_value[0] + self.real_param_vars['a22']*input_value[1]  + self.real_param_vars['b2']
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
    
    def real_param_variables(self):
        return self.real_param_vars
