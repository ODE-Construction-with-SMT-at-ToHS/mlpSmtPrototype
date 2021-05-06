from encoder import *
from z3 import *

import arguments

if __name__ == '__main__':
    
    #TODO Arguments are ignored for now
    args = arguments.parse_args()
    
    model_path = args.model
    myEncoder = Encoder(model_path)
    model_formula, result_vars = myEncoder.encode(0)

    #print(model_formula)
    print(result_vars)

    # Create a solver instance
    solver = Solver()

    # Assert subformulas
    for k, v in model_formula.items():
        solver.add(v)

    # Check for satisfiability
    res = solver.check()
    model = solver.model()

    print('The calculated result is: ')
    print(result_vars[0])
    print( model[result_vars[0]].as_decimal(4))
