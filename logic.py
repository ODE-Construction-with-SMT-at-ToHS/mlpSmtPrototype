from encoder import *
from z3 import *

def optimize_template(model_path, template, interval):
    """Function for optimizing parameters of a function-template
    to optimally fit a MLP.
    
    Attributes:
        model_path: path to export of keras model file.
        template: Template() instance # TODO has to be made compatible with args-parser
        range: tuple of tuples representing an intervall,
                limiting the function domain.
    """

    # Bounds
    lb, ub = interval
    # Tolerance
    epsilon = 5

    # Initial input
    x = lb

    # Encode the NN-model.
    myEncoder = Encoder(model_path)
    nn_model_formula, nn_output_vars, input_vars = myEncoder.encode()

    # Restore the actual NN.
    nn_model = keras.models.load_model(model_path)

    # Create a solver instance.
    solver_1 = Solver()

    cntr = 0
    
    # Encode 1. condition.
    nn_y = nn_model.predict([x])[0]
    t_output_vars = [Real('x1_{}_{}'.format(cntr,i)) for i in range(len(nn_output_vars))]
    formula_1 = []
    formula_1.append(template.generic_smt_encoding(x,t_output_vars))

    # norm 1 distance
    for i in range(len(nn_output_vars)):
        formula_1.append(nn_y[i] - t_output_vars[i] <= epsilon)
        formula_1.append(t_output_vars[i] - nn_y[i] <= epsilon)
    
    # Assert subformulas.
    for sub in formula_1:
        solver_1.add(sub)
    
    # Check for satisfiability.
    res = solver_1.check()
    fo_model = solver_1.model()
    if res == sat:
        print(fo_model)


def test_encoding(model_path, input):
    """Function that tests whether solving the encoding for a 
    MLP-model produces a correct model of the encoding.
    
    Attributes:
        model_path: path to export of keras model file.
        input: tuple representing the input.
    """

    # Encode the NN-model.
    myEncoder = Encoder(model_path)
    model_formula, result_vars, _ = myEncoder.encode()

    # Create a solver instance.
    solver = Solver()

    # Assert sub formulas.
    for k, v in model_formula.items():
        solver.add(v)

    # Encode the input.
    input_formula = myEncoder.encode_input(input)
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
        res_dec.append(fo_model[var].as_decimal(6))

    
    # Print the result for comparison.
    print('The calculated result is: ' + str(res_dec))
    print('However it should be:' + str(myEncoder.model.predict([input])))
