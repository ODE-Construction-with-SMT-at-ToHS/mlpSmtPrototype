from z3 import *
from tensorflow import keras

class Encoder():
    """
    Class for encoding Keras models as SMT formulas.
    """

    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)

        # Potential bug
        input_shape = self.model.input_shape 
        try:
            if (input_shape[2] != 1):
                print('Error: only the input_shape = [1] is supported currently.')
                print('Exiting ...')
                sys.exit()
            elif (self.model.__class__.__name__ != 'Sequential'):
                print('Error: only sequential models are supported currently.')
                print('Exiting ...')
                sys.exit()
        except:
                print('Error: only the input_shape = [1] is supported currently.')
                print('Exiting ...')
                sys.exit()
        
        self.weights = []
        for layer in self.model.layers:
            self.weights.append(layer.get_weights())

    def encode(self,x):
        """
        Encodes keras model as SMT forumla.
        """
        formula = {}

        # Create variables
        self.create_variables()
        print(self.variables_x)
        print(self.variables_y)

        # Encode affine layers
        formula['Affine layers:'] = self.encode_affine_layers()

        # Encode activation functions (relu_0)
        formula['Activation function:'] = self.encode_activation_function()

        # Encode input (currently only 1x1)
        formula['Input:'] = [self.variables_x[0][0] == x]

        result_vars = self.variables_x.pop()
        self.variables_x.append(result_vars)

        return formula, result_vars

    def create_variables(self):
        self.variables_x = []
        self.variables_y = []

        #TODO This has to be changed in order to allow other input shapes.
        # x_0_0 represents the input variable
        self.variables_x.append([Real('x_0_0')])
        
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

        # This currently only encodes relu_0
        for i in range(len(self.variables_y)):
            for j in range(len(self.variables_y[i])):
                function_encoding.append(Implies(0 >= self.variables_y[i][j], 
                    self.variables_x[i+1][j] == 0))
                function_encoding.append(Implies(0 < self.variables_y[i][j], 
                    self.variables_x[i+1][j] == self.variables_y[i][j]))

        return function_encoding
