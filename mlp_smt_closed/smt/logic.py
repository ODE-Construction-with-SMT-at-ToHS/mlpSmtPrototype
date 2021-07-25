import numpy as np

from mlp_smt_closed.smt.encoder import *
import multiprocessing
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Adaptor:
    """This class can be used to find a closed form for a trained MLP , given a function template.
    """

    def __init__(self, model_path, template, interval, splitting=False) -> None:
        """ Constructor. Encodes the model (/trained MLP) stored at ``model_path``, loads the model stored at
        ``model_path``, creates a ``z3`` solver instance. # TODO: comment on solver purpose

        
        Parameters:
            model_path: path to export of keras model file.
            template: Template() instance # TODO has to be made compatible with args-parser
            interval: tuple of tuples representing an interval, limiting the function domain. Within this domain, the
            output of the closed form and the MLP are not allowed to differ more than a (later specified) tolerance
        """

        # TODO add some sanity check to detect dimension errors early on.
        self.template = template
        self.model_path = model_path

        # lower and upper Bound
        # all values within this bound should be estimated within the epsilon-tolerance
        self.lb, self.ub = interval

        # Encode the NN-model.
        # call constructor, create Encoder instance
        self.my_encoder = Encoder(model_path)
        # encode the model
        self.splitting = splitting
        if not splitting:
            self.nn_model_formula, self.nn_output_vars, self.nn_input_vars = self.my_encoder.encode()
        else:
            self.nn_model_formulas, self.nn_output_vars, self.nn_input_vars = self.my_encoder.encode_splitted()
            # To make it pickleable
            self.nn_input_vars_as_str = [str(var) for var in self.nn_input_vars]

        # load the actual NN, used to calculate predictions
        self.nn_model = keras.models.load_model(model_path)

        # Create a solver instance. #TODO: why is solver_2 created first???, comment on purpose of this solver
        self.solver_2 = Solver()

    def adjust_template(self, epsilon: float = 0.5):
        """Method for finding parameters of a function-template to fit the MLP with maximal deviation ``epsilon``.
        TODO: describe difference to optimize template
        
        Parameters:
            epsilon = 0.5: Tolerance of template. Within the domain ``interval`` (specified at construction time), the
            output of the closed form and the MLP are not allowed to differ more than ``epsilon``
        """

        # Initial input
        x = self.lb
        print('Initial input: ', x)

        # Create a solver instance. This solver will be used to find parameters for the template within the epsilon
        # bound. In every iteration, one additional x value is added that needs to fit the constraints.
        solver_1 = Solver()

        # counts no. of iterations, used to define a new set of output variables in each iteration.
        counter = 0

        # initialize the result with "satisfiable"
        res = sat

        # while the encoding is satisfiable
        while res == sat:

            # Encode 1st condition, stored in formula_1:
            # the 1st condition is that the parameters found satisfy f(x) = NN(x) +- epsilon for all x currently
            # considered
            formula_1 = []

            # use NN to calculate output y for input x
            nn_y = self.nn_model.predict([x])[0]
            # create variables for each output value.
            t_output_vars = [FP(('y1_{}_{}'.format(counter, i)), Float32()) for i in range(len(self.nn_output_vars))]

            counter = counter + 1  # is it easier to understand this if its put at the end of the loop?

            # add encoding for function f according to template
            formula_1.append(self.template.generic_smt_encoding(x, t_output_vars))

            # add encoding to ensure output is within tolerance
            for i in range(len(self.nn_output_vars)):
                formula_1.append(float(nn_y[i]) - t_output_vars[i] <= epsilon)
                formula_1.append(t_output_vars[i] - float(nn_y[i]) <= epsilon)

            # Assert subformulas to solver.
            for sub in formula_1:
                solver_1.add(sub)
            
            # Check for satisfiability.
            print('Looking for new parameters')
            start_time_parameter = time.time()
            res = solver_1.check()
            if res == sat:
                fo_model = solver_1.model()
                # Update parameters.
                new_params = {}
                for key in self.template.get_params():
                    new_params[key] = get_float(fo_model, self.template.param_variables()[key])
                print('    -> New parameters found: ' + str(new_params))
                self.template.set_params(new_params)
            else:
                print('    -> Bound to strict: No parameters found.')
                break

            end_time_parameter = time.time()
            print('    -> took', end_time_parameter-start_time_parameter, 'seconds')

            # 2nd condition:
            print('Looking for new input')
            if not self.splitting:
                res, x = self._find_deviation(epsilon, refine=0)
            else:
                res, x = self._find_deviation_splitting(epsilon)

    def optimize_template(self):
        """Function for optimizing parameters of a function-template
        to fit an MLP optimally.
        """

        # Initial input
        x = self.lb
        print('Initial input: ', x)
        # Create a solver instances.
        solver_1 = Optimize()

        counter = 0
        prev_distance = None
        epsilon = None

        # initialize the result with "satisfiable"
        res = sat

        # while the encoding is satisfiable
        while res == sat:
            print('Optimizing parameters')
            start_time_opt = time.time()

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

            # print(formula_1)

            # Optimize parameters
            solver_1.minimize(distance)
            res = solver_1.check()
            if res == sat:
                fo_model = solver_1.model()
                # print(fo_model)

                # Update parameters.
                new_params = {}
                for key in self.template.get_params():
                    # print('Real: ' + str(fo_model.eval(self.template.real_param_variables()[key])))
                    new_params[key] = get_float(fo_model, self.template.real_param_variables()[key])
                print('    -> New parameters found: ' + str(new_params))
                self.template.set_params(new_params)

                # Optimal tolerance for the considered set of input values:
                new_epsilon = get_float(fo_model, distance)
                # TODO: I dont think this does something useful. --Nicolai
                if not (epsilon is None) and new_epsilon < epsilon:
                    # Casting issue from Real to pyhton value
                    print('Optimal parameters found.')
                    print('For a minimal deviation of: ' + str(epsilon))
                    break
                epsilon = new_epsilon
                print('    -> Deviation:', epsilon)
            else:
                print('Error! No satisfying parameters found.')
                print(res)
                break

            counter += 1

            end_time_opt = time.time()
            print('    -> Took', end_time_opt-start_time_opt, 'seconds')

            # Encode 2nd condition:
            print('Looking for new input')
            res, x = self._find_deviation(epsilon, refine=0)
            # print('End of while: ' + str(res)+str(x))

    def regression_verification_1d(self, epsilon: float = 0.5, size = 200):
        """Method for finding parameters of a function-template to fit the MLP with maximal deviation ``epsilon``.
        TODO: describe difference to optimize template

        Parameters:
            epsilon = 0.5: Tolerance of template. Within the domain ``interval`` (specified at construction time), the
            output of the closed form and the MLP are not allowed to differ more than ``epsilon``
        """
        start_time_regression_verification1d = time.time()

        # create samples form input network
        print('Taking samples to do regression')
        x_samples = np.linspace(self.lb, self.ub, size)
        y_samples = self.nn_model.predict(x_samples)


        # do regression to find parameters
        print('Doing regression')
        start_time_regression = time.time()
        reg = LinearRegression().fit(x_samples, y_samples)
        new_params = {'a': float(reg.coef_[0][0]), 'b': float(reg.intercept_[0])}
        self.template.set_params(new_params)

        print('    -> template parameters: ', self.template.get_params())
        print('    -> a = ', self.template.get_params()['a'])
        print('    -> b = ', self.template.get_params()['b'])
        print('    -> b rounded: ', round(self.template.get_params()['b'], 7))
        print('    -> type(a) = ', type(self.template.get_params()['a']))
        print('    -> type(b) = ', type(self.template.get_params()['b']))
        end_time_regression = time.time()
        print('    -> Function found: f(x) = ', reg.coef_[0][0], 'x +', reg.intercept_[0])
        print('    -> took', end_time_regression - start_time_regression, 'seconds')
        print(self.template.get_params())

        print('Looking for new input')
        if not self.splitting:
            res, x = self._find_deviation(epsilon, refine=0)
        else:
            res, x = self._find_deviation_splitting(epsilon)

        print(res)
        print(x)

        # new_params = {}
        # for key in self.template.get_params():
        #     new_params[key] = get_float(fo_model, self.template.param_variables()[key])
        # self.template.set_params(new_params)

        # while the encoding is satisfiable

        # do binary search to find close epsilon ---- or ---- directly do regression with max deviation as loss

        # Plot the results
        # plt.plot(x_samples, y_samples, 'r')
        plt.scatter(x_samples, y_samples)
        # plt.savefig('plots/' + func_class.name + '_learned.png')
        plt.show()
        plt.clf()





        # print('Looking for new input')
        # if not self.splitting:
        #     res, x = self._find_deviation(epsilon, refine=0)
        # else:
        #     res, x = self._find_deviation_splitting(epsilon)

    def _find_deviation(self, epsilon, refine=1):
        """
        This function is used to find an x value, for which |f(x) - NN(x)| relatively large. Therefore the function
        iteratively searches in for greater differences in each iteration. The central idea is that using x values for
        which the difference is large, will improve the parameter-estimation in the first part of the algorithm more
        than x values closer to the current estimation for f.

        Args:
            epsilon: distance in which new x values are searched initially. If found: doubled for next iteration
            refine: number of iterations of searching for new x-values
        Returns:
            (tuple):
                - bool to check whether a new value with minimum deviation ``epsilon`` is found
                - new x value, if found.,
        """
        # Encode the template.
        start_time_deviation = time.time()

        template_formula = self.template.smt_encoding()
        formula_2 = [template_formula]

        # Input conditions.
        for i in range(len(self.nn_input_vars)):
            formula_2.append(self.nn_input_vars[i] == self.template.input_variables()[i])
            formula_2.append(self.nn_input_vars[i] >= self.lb[i])
            formula_2.append(self.nn_input_vars[i] <= self.ub[i])

        # norm 1 distance TODO: combine with loop below
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
        # add NN constraints to solver
        formula_2.extend(self.nn_model_formula.values())
        # add constraints of the
        for sub in formula_2:
            self.solver_2.add(sub)

        # Create backtracking point for searching with greater epsilon
        # self.solver_2.push()
        self.solver_2.add(deviation)
        
        # Check for satisfiability. checking whether new x-value with minimum deviation epsilon exists
        print('    -> solving')
        res = self.solver_2.check()
        print('    -> solved')
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
                # print(deviation)
                self.solver_2.add(deviation)

                # Break the for loop, if no such input can be found
                print('    -> looking for improved input')
                if not self.solver_2.check():
                    print('    -> no improvement found')
                    break
                else:
                    print('    -> improvement found')

            fo_model = self.solver_2.model()
            # Extract new input for parameter correction.
            x_list = [get_float(fo_model, var) for var in self.nn_input_vars]
            x = tuple(x_list)
            print('    -> New input found: ' + str(x))
            end_time_deviation = time.time()
            print('    -> took', end_time_deviation-start_time_deviation, 'seconds')
            return res, x
        else:
            end_time_deviation = time.time()
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
            print('Most recent parameters sufficient (epsilon = ' + str(epsilon), ')')
            return res, None

    def _find_deviation_splitting(self, epsilon):
        start_time_deviation = time.time()
        # no incremental deviation
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
        formula_2.append(deviation)

        # Solve the splitted encoding in parallel
        print('    -> solving splits')
        processes = []
        result_q = multiprocessing.Queue()
        for split_formula in self.nn_model_formulas:
            split_formula.extend(formula_2)
            # This is neccesarry to be pickable
            formula_as_string = toSMT2Benchmark(And(split_formula), logic='QF_FPA')
            # input_vars_as_strings = [toSMT2Benchmark(var, logic='QF_FPA') for var in self.nn_input_vars]
            p = multiprocessing.Process(
                target=solve_single_split,
                args=(
                    formula_as_string,
                    self.nn_input_vars_as_str,
                    result_q)
                )
            processes.append(p)
            p.start()

        # Wait until one process returns sat, or all processes are done.
        waiting = True
        while waiting:
            # time.sleep(0.1)
            waiting = False
            for p in processes:
                if p.is_alive():
                    waiting = True
            if not result_q.empty():
                waiting = False

        # Terminate all processes, which are still alive.
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Pop the new input, if found.
        res, x = unsat, None
        if not result_q.empty():
            res, x = result_q.get()

        end_time_deviation = time.time()

        if res == sat:
            print('    -> New input found: ' + str(x))
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
        else:
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
            print('Most recent parameters sufficient (epsilon = ' + str(epsilon) + ')')
            print()

        return res, x

    def test_encoding(self, input):
        """Function that tests whether solving the encoding for a 
        MLP-model produces a correct model of the encoding.
        
        Parameters:
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
            # TODO make this cleaner?!
            res_dec.append(get_float(fo_model, var))

        # Print the result for comparison.
        print('The calculated result is: ' + str(res_dec))
        print('However it should be:' + str(self.my_encoder.model.predict([input])))


def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
    """Stolen from Stackoverflow
    not sure whats happening"""

    v = (Ast * 0)()
    return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())


def solve_single_split(formula_str, nn_input_vars, result):
    """Target function for solving splits
    """

    # Parse formula string
    formula = parse_smt2_string(formula_str)

    # Solve the formula
    solver = Solver()
    for subformula in formula:
        solver.add(subformula)
    print('        * solving single split')
    res = solver.check()
    print('        * single split solved: ' + str(res))

    # Extract the model, if sat
    if res == sat:
        # Preperation for extracting the new input
        # The order of the input vars has to be consistent
        x_list = [None for x in nn_input_vars]
        x_name_map = dict()
        for i in range(len(nn_input_vars)):
            x_name_map[nn_input_vars[i]] = i
        # This looses the order, but should speed up the look-up
        nn_input_vars = set(nn_input_vars)

        fo_model = solver.model()
        for var in fo_model:
            # This is a bit hacky, but not possible otherwise
            name = str(var)
            # print(name)
            if name in nn_input_vars:
                value = fo_model[var]
                float_value = float(eval(str(value)))
                x_list[x_name_map[name]] = float_value

        x = tuple(x_list)
        result.put((res, x))
