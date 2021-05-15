from operator import index
from z3 import *
from tensorflow import keras
import tensorflow as tf


class Encoder():
    """
    Class for encoding Keras models as SMT formulas.
    Only flat inputs and flat layers are supported.
    """

    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)

        try:
            # Potential bug
            _ , self.dimension_one = self.model.input_shape

            if (self.model.__class__.__name__ != 'Sequential'):
                print('Error: only sequential models are supported currently.')
                print('Exiting ...')
                sys.exit()
        except:
                print('Error: only flat input shapes (input_shape = (None, x)) is supported currently.')
                print('Exiting ...')
                sys.exit()
        
        self.weights = []
        for layer in self.model.layers:
            self.weights.append(layer.get_weights())

    def encode(self):
        """
        Encodes keras model as SMT forumla for an input x.
        """
        formula = {}

        # Create variables.
        self.create_variables()

        # Encode affine layers.
        formula['Affine layers'] = self.encode_affine_layers()

        # Encode activation functions (relu_0).
        formula['Activation function'] = self.encode_activation_function()

        # This specifies where to expect the output,input of the NN
        # in the model of an SMT-solver.
        output_vars = self.variables_x.pop()
        self.variables_x.append(output_vars)
        input_vars = self.variables_x[0]

        return formula, output_vars, input_vars

    def encode_input(self,x):
        """
        Encodes the value of the input.
        """

        input = []
        
        # x should be a list of a length corresponding to the input shape.
        if (len(x) != self.dimension_one):
            print('The input hast a wrong shape. Required: {} vs actual: {}'.format(len(x),self.dimension_one))
            print('Exiting ...')
            sys.exit()

        for j in range(self.dimension_one):
            input.append(self.variables_x[0][j] == x[j])
        
        return {'Input' : input}

    def create_variables(self):
        self.variables_x = []
        self.variables_y = []

        # Variables for the flat input. 
        self.variables_x.append([])
        for j in range(self.dimension_one):
            self.variables_x[0].append(Real('x_0_{}'.format(j)))
        
        # Variables x for the output of each layer.
        # Auxiliary variables y for applying the weights in each layer.
        for i in range(len(self.weights)):
            self.variables_x.append([])
            self.variables_y.append([])

            for j in range(len(self.weights[i][0][0])):
                self.variables_y[i].append(Real('y_{}_{}'.format(i,j)))
                self.variables_x[i+1].append(Real('x_{}_{}'.format(i+1,j)))

    # Encode affine layers
    def encode_affine_layers(self):
        affine_layers = []

        for i in range(len(self.variables_y)):
            for j in range(len(self.variables_y[i])):
                # Basically matrix multiplication
                # y_i_j = weights * ouput of last layer + bias
                affine_layers.append(self.variables_y[i][j] == 
                    Sum([(self.variables_x[i][j_x]*self.weights[i][0][j_x][j])
                    for j_x in range(len(self.variables_x[i]))])
                    + self.weights[i][1][j])
    
        return affine_layers
    
    def encode_activation_function(self):
        function_encoding = []

        # This currently only encodes relu_0 or linear
        for i in range(len(self.variables_y)):
            for j in range(len(self.variables_y[i])):
                # Determine which function to encode.
                activation = self.model.get_layer(index = i).activation.__name__ 
                if (activation == 'relu'):
                    function_encoding.append(Implies(0 >= self.variables_y[i][j], 
                        self.variables_x[i+1][j] == 0))
                    function_encoding.append(Implies(0 < self.variables_y[i][j], 
                        self.variables_x[i+1][j] == self.variables_y[i][j]))
                # This case also applies, if no activation is specified.
                elif (activation == 'linear'):
                    function_encoding.append(self.variables_x[i+1][j] == self.variables_y[i][j])
                
                else:
                    print('Error: only relu and linear are supported as activation function. Not: '
                        + str(activation))
                    print('Exiting ...')
                    sys.exit()
        
        return function_encoding
