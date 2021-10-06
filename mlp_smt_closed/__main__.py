"""This module contains the main function which starts our software depending on the arguments specified"""

from mlp_smt_closed.arguments import *
from mlp_smt_closed.smt.logic import *
from mlp_smt_closed.smt.templates import *
from mlp_smt_closed.mlp.functions import *

if __name__ == '__main__':

    # set encoding type
    encoding = 'Real'
    # encoding = 'FP'

    # parse arguements
    args = parse_args()

    # load keras model to get output dimension (necessary for template-choice, currently only n x n matrices
    # supported
    model = keras.models.load_model(args.model)
    output_dimension = model.layers[-1].output_shape[1]

    # distinguish between linear and polynomial template
    if args.template == 'linear':
        template = GenericLinearTemplate(encoding=encoding, n=output_dimension, m=output_dimension)
    # else: 'polynomial' was chosen
    else:
        template = PolynomialTemplate(degree=args.degree, variables=1)

    # init adaptor
    adaptor = Adaptor(args.model, template, args.intervals, splits=args.splits, encoding=encoding)

    # distinguish between smt and least squares
    if args.method == 'smt':
        # distinguish between adjust and optimize
        if args.optimize:
            adaptor.optimize_template()
        else:
            adaptor.adjust_template(epsilon = args.epsilon)
    # else: least squares (ls) was chosen
    else:
        # distiguish between linear and polynomial
        if args.template == 'linear':
            if output_dimension == 1:
                func_class = LinearA()
            elif output_dimension == 2:
                func_class = LinearA2D()
            elif output_dimension == 15:
                func_class = Platoon()
            else:
                print("Error: No support for this dimension yet. Try 'smt' instead of 'ls'.")
            adaptor.regression_verification_nd(func_class, args.sizes, args.epsilon, args.steps)
        # else: polynomial was chosen
        else:
            if args.degree == 1:
                func_class = LinearA()
            elif args.degree == 2:
                func_class = QuadraticA()
            elif args.degree == 3:
                func_class = PolyDeg3()
            else:
                print("Error: No support for this degree yet. Try 'smt' instead of 'ls'.")
            adaptor.polyfit_verification_1d(func_class, args.sizes, args.epsilon, args.steps, args.plot)
    
    '''
    # extract path to trained keras-model from arguments
    model_path = args.model

    start_time_overall = time.time()

    # set func_class, only necessary for if polyfit or regression used
    # func_class = PolyDeg3()

    # myLinTemplate = LinearTemplate(encoding=encoding)
    myLinTemplate = GenericLinearTemplate(encoding=encoding, n=2, m=2)
    # polyTemp = PolynomialTemplate(degree=func_class.degree(),variables=1)

    # just for debugging purposes
    # print(polyTemp.smt_encoding())
    # print(polyTemp.params)
    # print(polyTemp.generic_smt_encoding((3,),(Real('y'),)))
    # print(polyTemp.func((3,)))

    # init Adaptor
    # myAdaptor = Adaptor(model_path, myLinTemplate, ((-8,), (8,)), splits=0, encoding=encoding)
    # myAdaptor = Adaptor(model_path, polyTemp, ((-8,), (8,)), splits=0, encoding=encoding)
    myAdaptor = Adaptor(model_path, myLinTemplate, ((-8, -8), (8, 8)))

    # Test encoding
    # myAdaptor.test_encoding((42,))

    # find/verify parameters
    myAdaptor.adjust_template(epsilon = 0.05)
    # myAdaptor.optimize_template()
    # myAdaptor.polyfit_verification_1d(func_class, [200], 0.5, 4, args.plot)
    # myAdaptor.regression_verification_1d()
    # myAdaptor.regression_verification_nd(func_class, [101, 101], 0.5, 4)
    
    end_time_overall = time.time()

    print('Overall computation time: ', end_time_overall - start_time_overall, 'seconds')

    # Test template optimization
    # myAdaptor.optimize_template()
    '''
    '''# LinearA2D was trained on ((-10,-10),(10,10))
    myLinTemplate = Linear2DTemplate()
    path_to_model = os.path.join('mlp_smt_closed','mlp', 'models', 'Linear2DA_model.h5')
    myAdaptor = Adaptor(path_to_model, myLinTemplate, ((-10,-10),(10,10)))

    # Test en ding
    #myAdaptor.test_encoding((1,1))

    # Test template adjustment
    myAdaptor.adjust_template(epsilon=1)

    # Test template optimization
    #myAdaptor.optimize_template()

    # Brusselator was trained on ((-2,2),(-2,2))

    print('Total time: '+ str(time.time()-start_time))'''
