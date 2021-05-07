from encoder import *
from z3 import *

import arguments

if __name__ == '__main__':
    
    #TODO Arguments are ignored for now
    args = arguments.parse_args()
    
    x = 0

    model_path = args.model
    myEncoder = Encoder(model_path)
    model_formula, result_vars = myEncoder.encode(x)

    #print(model_formula)
    print(result_vars)

    # Create a solver instance
    solver = Solver()

    # Assert subformulas
    for k, v in model_formula.items():
        solver.add(v)

    # Check for satisfiability
    res = solver.check()
    fo_model = solver.model()

    print('The calculated result is: ')
    print(model_formula)
    print(result_vars[0])
    print(res)
    print(fo_model[result_vars[0]].as_decimal(4))
    print('However it should be:')
    print(myEncoder.model.predict([x]))
