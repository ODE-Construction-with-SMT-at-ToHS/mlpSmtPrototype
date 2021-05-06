from z3 import *
from tensorflow import keras

class Encoder():
    """
    Class for encoding Keras models as SMT formulas.
    """

    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)
        self.weights = []
        for layer in self.model.layers:
            self.weights.append(layer.get_weights())

    def encode(self,x):
        """
        Encodes keras model as SMT forumla.
        """

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

        #Kreate variables
        self.create_variables()
        print(self.variables_x)
        print(self.variables_y)


        return 'Nothing to see here.'

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
            

            
        pass
