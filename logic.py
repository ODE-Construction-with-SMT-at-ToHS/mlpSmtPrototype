from encoder import *
from z3 import *


def optimize_template(model_path, template, interval, epsilon: float=0.5):
    """Function for optimizing parameters of a function-template
    to optimally fit an MLP.
    
    Parameters:
        model_path: path to export of keras model file.
        template: Template() instance # TODO has to be made compatible with args-parser
        range: tuple of tuples representing an interval,
                limiting the function domain.
        epsilon = 5: Tolerance of template.
    """

    # TODO add some sanity check to detect dimension errors early on.

    # lower and upper Bound
    # all values within this bound should be estimated within the epsilon-tolerance
    lb, ub = interval

    # Initial input
    x = lb

    # Encode the NN-model.
    my_encoder = Encoder(model_path)
    nn_model_formula, nn_output_vars, nn_input_vars = my_encoder.encode()

    # Restore the actual NN.
    nn_model = keras.models.load_model(model_path)

    # Create a solver instances.
    solver_1 = Solver()
    solver_2 = Solver()

    counter = 0

    # initialize the result with "satisfiable"
    res = sat

    # while the encoding is satisfiable
    while res == sat:

        # Encode 1st condition, stored in formula_1:
        # used to check whether there exists a function f such that f(x) = NN(x) +- epsilon
        formula_1 = []

        # use NN to calculate output y for input x
        nn_y = nn_model.predict([x])[0]
        # create variables for each output value.
        t_output_vars = [FP(('y1_{}_{}'.format(counter, i)), Float32()) for i in range(len(nn_output_vars))]

        counter = counter + 1  # is it easier to understand this if its put at the end of the loop?

        # add encoding for function f according to template
        formula_1.append(template.generic_smt_encoding(x, t_output_vars))

        # ensure output is within tolerance
        for i in range(len(nn_output_vars)):
            formula_1.append(float(nn_y[i]) - t_output_vars[i] <= epsilon)
            formula_1.append(t_output_vars[i] - float(nn_y[i]) <= epsilon)
        
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
                new_params[key] = get_float(fo_model, template.param_variables()[key])
            print(new_params)
            template.set_params(new_params)
        else:
            print('No parameters within bound found.')
            break

        # Encode 2nd condition:

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
                           )
                         )
        
        print(formula_2)
        # Assert subformulas.
        solver_2.reset()
        for sub in formula_2:
            solver_2.add(sub)
        for _, nn_encoding_constraint in nn_model_formula.items():
            solver_2.add(nn_encoding_constraint)
        
        # Check for satisfiability.
        res = solver_2.check()
        if res == sat:
            fo_model = solver_2.model()
            # Extract new input for parameter correction.
            x_list = [get_float(fo_model, var) for var in nn_input_vars]
            x = tuple(x_list)
            print('New input: ' + str(x))
        else:
            print('Parameters within bound found.')


def test_encoding(model_path, input):
    """Function that tests whether solving the encoding for a 
    MLP-model produces a correct model of the encoding.
    
    Parameters:
        model_path: path to export of keras model file.
        input: tuple representing the input.
    """

    # Encode the NN-model.
    my_encoder = Encoder(model_path)
    model_formula, result_vars, _ = my_encoder.encode()

    # Create a solver instance.
    solver = Solver()

    # Assert sub formulas.
    for k, v in model_formula.items():
        solver.add(v)

    # Encode the input.
    input_formula = my_encoder.encode_input(input)
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
        # This is a suspiciously hacky solution.
        #TODO make this cleaner?! 
        res_dec.append(get_float(fo_model, var))

    # Print the result for comparison.
    print('The calculated result is: ' + str(res_dec))
    print('However it should be:' + str(my_encoder.model.predict([input])))
