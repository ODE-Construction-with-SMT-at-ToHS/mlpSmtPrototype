from encoder import *
from z3 import *

def optimize_template(model_path, template, interval, epsilon = 1):
    """Function for optimizing parameters of a function-template
    to optimally fit a MLP.
    
    Attributes:
        model_path: path to export of keras model file.
        template: Template() instance # TODO has to be made compatible with args-parser
        range: tuple of tuples representing an intervall,
                limiting the function domain.
        epsilon = 5: Tolerance of template.
    """

    # TODO add some sanity check to detect dimension errors early on.

    # Bounds
    lb, ub = interval

    # Initial input
    x = lb

    # Encode the NN-model.
    myEncoder = Encoder(model_path)
    nn_model_formula, nn_output_vars, nn_input_vars = myEncoder.encode()

    # Restore the actual NN.
    nn_model = keras.models.load_model(model_path)

    # Create a solver instances.
    solver_1 = Solver()
    solver_2 = Solver()

    cntr = 0
    res = sat

    while (res == sat):

        # Encode 1. condition.
        nn_y = nn_model.predict([x])[0]
        # We need fresh output variables for each new, x value.
        t_output_vars = [Real('y1_{}_{}'.format(cntr,i)) for i in range(len(nn_output_vars))]
        cntr = cntr +1
        formula_1 = []
        formula_1.append(template.generic_smt_encoding(x,t_output_vars))

        # norm 1 distance
        for i in range(len(nn_output_vars)):
            formula_1.append(nn_y[i] - t_output_vars[i] <= epsilon)
            formula_1.append(t_output_vars[i] - nn_y[i] <= epsilon)
        
        print(formula_1)

        # Assert subformulas.
        for sub in formula_1:
            solver_1.add(sub)
        
        # Check for satisfiability.
        res = solver_1.check()
        if res == sat:
            fo_model = solver_1.model()
            print(fo_model)
            # Update parameters.
            new_params = {}
            for key in template.get_params():
                new_params[key] = fo_model[template.param_variables()[key]]
            print(new_params)
            template.set_params(new_params)
        else:
            print('No parameters within bound found.')
            break
            
        
        # Encode 2. condition

        # Encode the template.
        template_formula = template.smt_encoding()
        formula_2 = [template_formula]

        # Input conditions.
        for i in range(len(nn_input_vars)):
            formula_2.append(nn_input_vars[i] == template.input_variables()[i])
            formula_2.append(nn_input_vars[i] >= lb[i])
            formula_2.append(nn_input_vars[i] <= ub[i])

        # norm 1 distance
        formula_2.append(Or(
                Or(
                    [nn_output_vars[i] - template.output_variables()[i] > epsilon 
                    for i in range(len(nn_output_vars))]),
                Or(
                    [template.output_variables()[i] - nn_output_vars[i] > epsilon 
                    for i in range(len(nn_output_vars))])
                ))
        
        print(formula_2)
        # Assert subformulas.
        solver_2.reset()
        for sub in formula_2:
            solver_2.add(sub)
        for _,v in nn_model_formula.items():
            solver_2.add(v)
        
        # Check for satisfiability.
        res = solver_2.check()
        if res == sat:
            fo_model = solver_2.model()
            # Extract new input for parameter correction.
            # This causes precision loss, but is necessary to be suitable as 
            # input for the NN in the next iteration.
            x_list = [float(fo_model[var].as_fraction().numerator)/float(fo_model[var].as_fraction().denominator) for var in nn_input_vars]
            x = tuple(x_list)
            print('New input: ' + str(x))
        else:
            print('Parameters within bound found.')



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
