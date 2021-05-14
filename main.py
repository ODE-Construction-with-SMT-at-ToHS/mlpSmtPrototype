from logging import log
import arguments
import logic
from  templates import *

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = arguments.parse_args()
    
    model_path = args.model

    # Test encoding
    #logic.test_encoding(model_path,(42,))

    myLinTemplate = LinearTemplate()
    logic.optimize_template(model_path, myLinTemplate, ((-10,),(10,)) )
