from logging import log
import arguments
import smt.logic
from smt.templates import *

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = arguments.parse_args()

    # extract path to trained keras-model from arguments
    model_path = args.model

    # Test encoding
    #logic.test_encoding(model_path,(42,))

    myLinTemplate = LinearTemplate()

    # Test template adjustment
    #smt.logic.adjust_template(model_path, myLinTemplate, ((-8,), (8,)))

    # Test template optimization
    smt.logic.optimize_template(model_path, myLinTemplate, ((-8,), (8,)))
