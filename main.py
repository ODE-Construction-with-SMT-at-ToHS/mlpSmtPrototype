from logging import log
import arguments
import logic
from templates import *

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = arguments.parse_args()

    # extract path to trained keras-model from arguments
    model_path = args.model

    # Test encoding
    #logic.test_encoding(model_path,(42,))

    # Test template optimization
    myLinTemplate = LinearTemplate()
    logic.optimize_template(model_path, myLinTemplate, ((-8,), (8,)))
