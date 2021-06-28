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

    myLinTemplate = LinearTemplate()
    myAdaptor = Adaptor(model_path, myLinTemplate, ((-8,), (8,)))
    # LinearA2D was trained on ((-10,10),(-10,10))
    # Brusselator was trained on ((-2,2),(-2,2))

    start_time = time.time()

    # Test encoding
    #myAdaptor.test_encoding((42,))

    # Test template adjustment
    #myAdaptor.adjust_template()

    # Test template optimization
    myAdaptor.optimize_template()

    print('Total time: '+ str(time.time()-start_time))
