from encoder import *
from z3 import *

import arguments

if __name__ == '__main__':
    
    # TODO Arguments are ignored for now
    args = arguments.parse_args()

    # Encode the model.
    model_path = args.model
    myEncoder = Encoder(model_path)
    model_formula, result_vars = myEncoder.encode()

    # Create a solver instance.
    solver = Solver()

    # Assert sub formulas.
    for k, v in model_formula.items():
        solver.add(v)

    # Encode some input.
    # All inputs must be tupels!
    x = (4,3)
    print(len(x))
    input_formula = myEncoder.encode_input(x)
    for k, v in input_formula.items():
        solver.add(v)

    # Check for satisfiability.
    res = solver.check()
    fo_model = solver.model()
    if res != sat:
        print('ERROR. Formula is not satisfiable.')
        sys.exit()

    # Convert to readable decimal representation.
    res_dec = []
    for var in result_vars:
        res_dec.append(fo_model[var].as_decimal(4))

    
    # Print the result for comparison.
    print('The calculated result is: ' + str(res_dec))
    print('However it should be:' + str(myEncoder.model.predict([x])))
