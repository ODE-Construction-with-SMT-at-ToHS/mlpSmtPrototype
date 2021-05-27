from z3 import *
from tensorflow import keras

import functools


class Encoder:
    """
    Class for encoding Keras models as SMT formulas.
    Only flat inputs and flat layers are supported.
    """

    # constructor
    def __init__(self, modelpath):
        # load model stored at modelpath
        self.model = keras.models.load_model(modelpath)

        # check whether model sequential and ??? (what does the exception do?)
        try:
            # Potential bug
            _, self.dimension_one = self.model.input_shape

            if self.model.__class__.__name__ != 'Sequential':
                print('Error: only sequential models are supported currently.')
                print('Exiting ...')
                sys.exit()
        except:
            print('Error: only flat input shapes (input_shape = (None, x)) is supported currently.')
            print('Exiting ...')
            sys.exit()

        """get and reshape weights in all layers.
           form for layer n with i nodes and layer n+1 with j nodes: 
           array([[<weight_1->1>,...,<weight_1->j>],...,[<weight_i->1>,...,<weight_i->j>]], dtype = <type>),
           array([<bias_1>, ..., <bias_2>], dtype = <type>)
        """
        self.weights = [layer.get_weights() for layer in self.model.layers]

    def encode(self):
        """
        Encodes keras model as SMT formula for an input x.
        """
        formula = {}

        # Create variables.
        self._create_variables()

        # Encode affine layers (affine layers = application of weights and bias).
        formula['Affine layers'] = self._encode_affine_layers()

        # Encode activation functions (relu_0).
        formula['Activation function'] = self._encode_activation_function()

        # This specifies where to expect the output,input of the NN
        # in the model of an SMT-solver.
        input_vars = self.variables_x[0]
        output_vars = self.variables_x[-1]

        return formula, output_vars, input_vars

    def encode_input(self, x):
        """
        Encodes the value of the input.
        """

        input_value = []
        
        # x should be a list of a length corresponding to the input shape.
        if len(x) != self.dimension_one:
            print('The input hast a wrong shape. Required: {} vs actual: {}'.format(len(x), self.dimension_one))
            print('Exiting ...')
            sys.exit()

        for j in range(self.dimension_one):
            input_value.append(self.variables_x[0][j] == x[j])
        
        return {'Input': input_value}

    def _create_variables(self):
        # one array entry for each layer, one additional input layer.
        # Variables x for the output of each layer (after applying ReLU).
        # Auxiliary variables y for applying the weights in each layer (after applying weights and bias).
        self.variables_x = []
        self.variables_y = []

        # create variables for the (flat) input layer.
        self.variables_x.append([])
        for j in range(self.dimension_one):
            self.variables_x[0].append(FP('x_0_{}'.format(j), Float32()))
        
        # create variables for the actual layers
        # iterate over the layers
        for i in range(len(self.weights)):
            self.variables_x.append([])
            self.variables_y.append([])
            # iterate over nodes of one layer
            for j in range(len(self.weights[i][0][0])):
                # y-var for output after applying weights+bias
                self.variables_y[i].append(FP('y_{}_{}'.format(i, j), Float32()))
                # x-var for output of layer (after applying ReLU)
                self.variables_x[i+1].append(FP('x_{}_{}'.format(i+1, j), Float32()))

    # Encode affine layers
    def _encode_affine_layers(self):
        affine_layers = []

        # iterate over each layer
        for i in range(len(self.variables_y)):
            # iterate over each node of layer i
            for j in range(len(self.variables_y[i])):
                # Basically matrix multiplication
                # y_i_j = weights * output of last layer + bias
                # the equation ("==") is appended as a constraint for a solver
                affine_layers.append(
                    self.variables_y[i][j] == 
                    gen_sum([(self.variables_x[i][j_x]* 
                        float(self.weights[i][0][j_x][j]))
                        for j_x in range(len(self.variables_x[i]))])
                    + float(self.weights[i][1][j]))
    
        return affine_layers
    
    def _encode_activation_function(self):
        function_encoding = []

        # This currently only encodes relu_0 or linear
        # iterate over layers
        for i in range(len(self.variables_y)):
            # iterate over variables of the layers
            for j in range(len(self.variables_y[i])):
                # Determine which function to encode.
                activation = self.model.get_layer(index=i).activation.__name__
                # encode ReLU (ReLU(x) = { 0, 0 >= x
                #                          x, else   }
                if activation == 'relu':
                    function_encoding.append(Implies(0 >= self.variables_y[i][j], 
                                             self.variables_x[i+1][j] == 0))
                    function_encoding.append(Implies(0 < self.variables_y[i][j], 
                                                     self.variables_x[i+1][j] == self.variables_y[i][j]))
                # This case also applies, if no activation is specified.
                elif activation == 'linear':
                    function_encoding.append(self.variables_x[i+1][j] == self.variables_y[i][j])
                
                else:
                    print('Error: only relu and linear are supported as activation function. Not: '
                          + str(activation))
                    print('Exiting ...')
                    sys.exit()
        
        return function_encoding


# Creates a sum of all elements in the list.
def gen_sum(lst):
    return functools.reduce(lambda a, b: a+b, lst, 0)


# Converts FP to float
def get_float(fo_model, var):
    # This is a suspiciously hacky solution.
    #TODO make this cleaner?! 
    return float(eval(str(fo_model.eval(var,model_completion=True))))
