import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from logging import log
from mlp_smt_closed.arguments import *
from mlp_smt_closed.smt.logic import *
from mlp_smt_closed.smt.templates import *

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = parse_args()

    # extract path to trained keras-model from arguments
    model_path = args.model

    start_time_overall = time.time()

    encoding = 'Real'
    #encoding = 'FP'
    myLinTemplate = LinearTemplate(encoding=encoding)
    myAdaptor = Adaptor(model_path, myLinTemplate, ((-8,), (8,)), splits=0, encoding=encoding)

    # Test encoding
    #myAdaptor.test_encoding((42,))

    # Test template adjustment
    myAdaptor.adjust_template(epsilon = 0.2)
    # myAdaptor.optimize_template()

    end_time_overall = time.time()

    print('Overall computation time: ', end_time_overall - start_time_overall, 'seconds')

    # Test template optimization
    #myAdaptor.optimize_template()

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
