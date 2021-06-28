from itertools import count
from math import dist
from z3 import *
from mlp_smt_closed.smt.encoder import *

class Adaptor:

    def __init__(self, model_path, template, interval) -> None:
        """Class for finding parameters of a function-template
        to fit an MLP.
        
        Parameters:
            model_path: path to export of keras model file.
            template: Template() instance # TODO has to be made compatible with args-parser
            interval: tuple of tuples representing an interval,
                    limiting the function domain.
        """

        # TODO add some sanity check to detect dimension errors early on.
        self.template = template
        self.model_path = model_path

        # lower and upper Bound
        # all values within this bound should be estimated within the epsilon-tolerance
        self.lb, self.ub = interval

        # Encode the NN-model.
        self.my_encoder = Encoder(model_path)
        self.nn_model_formula, self.nn_output_vars, self.nn_input_vars = self.my_encoder.encode()

        # Restore the actual NN.
        self.nn_model = keras.models.load_model(model_path)

        # Create a solver instances.
        self.solver_2 = Solver()

    def adjust_template(self, epsilon: float=0.5):
        """Function for finding parameters of a function-template
        to fit an MLP with a specified maximal deviation.
        
        Parameters:
            epsilon = 0.5: Tolerance of template.
        """

        # Initial input
        x = self.lb

        # Create a solver instance.
        solver_1 = Solver()

        counter = 0

        # initialize the result with "satisfiable"
        res = sat

        # while the encoding is satisfiable
        while res == sat:

            # Encode 1st condition, stored in formula_1:
            # used to check whether there exists a function f such that f(x) = NN(x) +- epsilon
            formula_1 = []

            # use NN to calculate output y for input x
            nn_y = self.nn_model.predict([x])[0]
            # create variables for each output value.
            t_output_vars = [FP(('y1_{}_{}'.format(counter, i)), Float32()) for i in range(len(self.nn_output_vars))]

            counter = counter + 1  # is it easier to understand this if its put at the end of the loop?

            # add encoding for function f according to template
            formula_1.append(self.template.generic_smt_encoding(x, t_output_vars))

            # ensure output is within tolerance
            for i in range(len(self.nn_output_vars)):
                formula_1.append(float(nn_y[i]) - t_output_vars[i] <= epsilon)
                formula_1.append(t_output_vars[i] - float(nn_y[i]) <= epsilon)

            # Assert subformulas.
            for sub in formula_1:
                solver_1.add(sub)
            
            # Check for satisfiability.
            res = solver_1.check()
            if res == sat:
                fo_model = solver_1.model()
                # Update parameters.
                new_params = {}
                for key in self.template.get_params():
                    new_params[key] = get_float(fo_model, self.template.param_variables()[key])
                print('New parameters: ' + str(new_params))
                self.template.set_params(new_params)
            else:
                print('No parameters within bound found.')
                break

            # 2nd condition:
            res, x = self._find_deviation(epsilon, refine=0)

    def optimize_template(self):
        """Function for optimizing parameters of a function-template
        to fit an MLP optimally.
        """

        # Initial input
        x = self.lb

        # Create a solver instances.
        solver_1 = Optimize()

        counter = 0
        prev_distance = None
        epsilon = None

        # initialize the result with "satisfiable"
        res = sat

        # while the encoding is satisfiable
        while res == sat:

            # Encode 1st condition, stored in formula_1:
            # used to check whether there exists a function f such that f(x) = NN(x) +- epsilon
            formula_1 = []

            # use NN to calculate output y for input x
            nn_y = self.nn_model.predict([x])[0]
            # create variables for each output value.
            t_output_vars = [Real('y1_{}_{}'.format(counter, i)) for i in range(len(self.nn_output_vars))]

            # add encoding for function f according to template
            formula_1.append(self.template.generic_real_smt_encoding(x, t_output_vars))

            # Define the deviation incrementally

            # Firstly, encode the distance in each dimension
            absolutes = [Real('abs_{}_{}'.format(counter, i)) for i in range(len(self.nn_output_vars))]
            for i in range(len(self.nn_output_vars)):
                formula_1.append(absolutes[i] ==
                    If(float(nn_y[i]) - t_output_vars[i] >= 0,
                        float(nn_y[i]) - t_output_vars[i],
                        t_output_vars[i] - float(nn_y[i])
                    )
                )
            
            # Secondly, sum up those distances
            sum_abs = gen_sum(absolutes)
            distance = Real('distance_{}'.format(counter))
            if prev_distance is None:
                dist_enc = distance == sum_abs
            else:
                # TODO check whether runtime improves if prev_distance is
                # not stored, but instead push(), pop() are used with a
                # new sum over all inputs in each iteration.
                dist_enc = distance == sum_abs + prev_distance
                prev_distance = sum_abs
                
                # pop prev_distance
                solver_1.pop()

            # Assert subformulas.
            for sub in formula_1:
                solver_1.add(sub)
            solver_1.push()
            solver_1.add(dist_enc)

            print(formula_1)
            
            # Optimize parameters
            solver_1.minimize(distance)
            res = solver_1.check()
            if res == sat:
                fo_model = solver_1.model()
                #print(fo_model)

                # Update parameters.
                new_params = {}
                for key in self.template.get_params():
                    print('Real: ' + str(fo_model.eval(self.template.real_param_variables()[key])))
                    new_params[key] = get_float(fo_model, self.template.real_param_variables()[key])
                print('New parameters: ' + str(new_params))
                self.template.set_params(new_params)

                # Optimal tolerance for the considered set of input values:
                new_epsilon = get_float(fo_model, distance)
                if not (epsilon is None) and new_epsilon < epsilon:
                    # Casting issue from Real to pyhton value
                    print('Optimal parameters found.')
                    print('For a minimal deviation of: ' + str(epsilon))
                    break
                epsilon = new_epsilon
                print('With a deviation of: ' + str(epsilon))
            else:
                print('Error! No satisfying parameters found.')
                print(res)
                break

            counter += 1

            # Encode 2nd condition:
            res, x = self._find_deviation(epsilon, refine=0)
            print('End of while: ' + str(res)+str(x))

    def _find_deviation(self, epsilon, refine=1):
        # Encode the template.
        template_formula = self.template.smt_encoding()
        formula_2 = [template_formula]

        # Input conditions.
        for i in range(len(self.nn_input_vars)):
            formula_2.append(self.nn_input_vars[i] == self.template.input_variables()[i])
            formula_2.append(self.nn_input_vars[i] >= self.lb[i])
            formula_2.append(self.nn_input_vars[i] <= self.ub[i])

        # norm 1 distance
        deviation = Or(
                            Or(
                            [self.nn_output_vars[i] - self.template.output_variables()[i] > epsilon
                                for i in range(len(self.nn_output_vars))]),
                            Or(
                            [self.template.output_variables()[i] - self.nn_output_vars[i] > epsilon
                                for i in range(len(self.nn_output_vars))])
                        )
        
        # Assert subformulas.
        self.solver_2.reset()
        formula_2.extend(self.nn_model_formula.values())
        for sub in formula_2:
            self.solver_2.add(sub)

        # Create backtracking point for searching with greater epsilon
        #self.solver_2.push()
        self.solver_2.add(deviation)
        
        # Check for satisfiability.
        print('checking violation')
        res = self.solver_2.check()
        print('done checking')
        if res == sat:

            # Heuristically search for a greater deviation
            exp_eps = epsilon
            for _ in range(refine):

                # Incremental solver does not seem to work
                self.solver_2.reset()
                for sub in formula_2:
                    self.solver_2.add(sub)

                # Double epsilon and encode it
                exp_eps = exp_eps*2
                deviation = Or(
                    Or(
                        [self.nn_output_vars[i] - self.template.output_variables()[i] > exp_eps
                        for i in range(len(self.nn_output_vars))]),
                    Or(
                        [self.template.output_variables()[i] - self.nn_output_vars[i] > exp_eps
                        for i in range(len(self.nn_output_vars))])
                    )
                
                print(deviation)
                self.solver_2.add(deviation)

                # Break the for loop, if no such input can be found
                if not self.solver_2.check():
                    break
                else:
                    print('Found better new input.')

            fo_model = self.solver_2.model()
            # Extract new input for parameter correction.
            x_list = [get_float(fo_model, var) for var in self.nn_input_vars]
            x = tuple(x_list)
            print('New input: ' + str(x))
            return res, x
        else:
            print('Parameters found.')
            print('With epsilon = ' + str(epsilon))
            return res, None

    def test_encoding(self, input):
        """Function that tests whether solving the encoding for a 
        MLP-model produces a correct model of the encoding.
        
        Parameters:
            model_path: path to export of keras model file.
            input: tuple representing the input.
        """

        # Create a solver instance.
        solver = Solver()

        # Assert sub formulas.
        for k, v in self.nn_model_formula.items():
            solver.add(v)

        # Encode the input.
        input_formula = self.my_encoder.encode_input(input)
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
        for var in self.nn_output_vars:
            # This is a suspiciously hacky solution.
            #TODO make this cleaner?! 
            res_dec.append(get_float(fo_model, var))

        # Print the result for comparison.
        print('The calculated result is: ' + str(res_dec))
        print('However it should be:' + str(self.my_encoder.model.predict([input])))
