from logging import log
from mlp_smt_closed.arguments import *
from mlp_smt_closed.smt.logic import *
from mlp_smt_closed.smt.templates import *

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = parse_args()

    # extract path to trained keras-model from arguments
    model_path = args.model

    # Test encoding
    #logic.test_encoding(model_path,(42,))

    myLinTemplate = LinearTemplate()

    # Test template adjustment
    #smt.logic.adjust_template(model_path, myLinTemplate, ((-8,), (8,)))

    # Test template optimization
    optimize_template(model_path, myLinTemplate, ((-8,), (8,)))
