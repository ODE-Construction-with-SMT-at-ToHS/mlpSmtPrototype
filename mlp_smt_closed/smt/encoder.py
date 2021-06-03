"""This module contains a class to encode Keras models in ``z3`` as well as two helper functions"""

from z3 import *
from tensorflow import keras

import functools


class Encoder:
    """
    Class for encoding Keras models as SMT formulas using ``z3``. Only flat inputs and flat layers are supported.
    The encoding is done as described in `this paper <https://arxiv.org/abs/2008.01204>`_.
    """

    def __init__(self, modelpath):
        """
        Constructor. Check whether a model is stored at ``modelpath``, it is sequential, and the input is
        flat-shaped. Also, the model is loaded, reshaped and saved in ``self.weights``

        Args:
            modelpath (path):
                path to the file in which the model that should be encoded is stored.
        """
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

        # get and reshape weights in all layers.
        # form for layer n with i nodes and layer n+1 with j nodes:
        # array([[<weight_1->1>,...,<weight_1->j>],...,[<weight_i->1>,...,<weight_i->j>]], dtype = <type>),
        # array([<bias_1>, ..., <bias_2>], dtype = <type>)
        self.weights = [layer.get_weights() for layer in self.model.layers]

    def encode(self):
        """
        Encode Keras model saved at ``modelpath`` as SMT formula in ``z3`` for a variable input w.r.t. the MLP the model
        encodes. We used the encoding described `here <https://arxiv.org/abs/2008.01204>`_.

        Returns:
            (tuple): tuple containing:
                - formula: ``z3`` formula encoding the model
                - output_vars: list of variables representing the output of the model
                - input_vars: list of variables representing the input of the model
        """
        formula = {}

        # Create variables.
        self._create_variables()

        # Encode affine layers (affine layers = application of weights and bias).
        formula['Affine layers'] = self._encode_affine_layers()

        # Encode activation functions (relu_0).
        formula['Activation function'] = self._encode_activation_function()

        # This specifies where to expect the output, input of the NN
        # in the model of an SMT-solver.
        input_vars = self.variables_x[0]
        output_vars = self.variables_x[-1]

        return formula, output_vars, input_vars

    def encode_input(self, x):
        """
        Assign ``x`` to the corresponding input variable
        TODO: document return
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
        """
        Creates two ``z3`` variables for each node in every layer, plus additional variables for the input layer.
        The first variable contains the value after applying weights and bias, the second variable contains the value
        after applying the activation function (which also is the output of the corresponding node).
        """
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

    # Encode affine functions
    def _encode_affine_layers(self):
        """
        Encodes affine functions through adding a linear relationship between the ``x`` and ``y`` variable for each
        node. This relationship is fully defined by the weights and biases from the Keras model.

        Returns:
            (list): List of constraints ensuring correct encoding of the affine functions
        """
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
        """
        Encodes activation function given by the Keras model for each node in every layer.

        Returns:
            (list): List of constraints ensuring correct encoding of the activation functions
        """
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


def gen_sum(list):
    """Creates a sum of all elements in the list."""
    return functools.reduce(lambda a, b: a+b, list, 0)


def get_float(fo_model, var):
    """Converts z3 value to python float."""
    # This is a suspiciously hacky solution.
    # TODO make this cleaner?!
    return float(eval(str(fo_model.eval(var, model_completion=True))))
