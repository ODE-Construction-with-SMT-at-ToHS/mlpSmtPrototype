from os import getpid
from re import template
import numpy as np

from mlp_smt_closed.smt.encoder import *
from mlp_smt_closed.smt.templates import *
import multiprocessing, threading
import time, sys
import matplotlib.pyplot as plt
from numpy.lib.shape_base import split
from sklearn.linear_model import LinearRegression


class Adaptor:
    """This class can be used to find a closed form for a trained MLP , given a function template.
    """

    def __init__(self, model_path, template, interval, splits=0, encoding='Real') -> None:
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
        self.my_encoder = Encoder(model_path, encoding=encoding)
        # encode the model
        self.splits = splits
        self.encoding = encoding

        if splits == 0:
            self.nn_model_formula, self.nn_output_vars, self.nn_input_vars = self.my_encoder.encode()
        else:
            self.nn_model_formula = None
            self.nn_model_formulas, self.nn_output_vars, self.nn_input_vars = self.my_encoder.encode_splitted(number=splits)

            # Multiprocessing preparation.
            self.jobs = multiprocessing.Queue()
            self.solutions = multiprocessing.Queue()

            self.processes = []
            self._init_worker_solvers()

        # load the actual NN, used to calculate predictions
        self.nn_model = keras.models.load_model(model_path)

        # Create a solver instance. #TODO: why is solver_2 created first???, comment on purpose of this solver
        self.solver_2 = Solver()

    def adjust_template(self, epsilon = 0.5):
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

        # Float precision for real encoding
        precision = Q(1,10000)
        param_vars = self.template.param_variables()
        prec_formula = []

        # while the encoding is satisfiable
        while res == sat:

            # Encode 1st condition, stored in formula_1:
            # the 1st condition is that the parameters found satisfy f(x) = NN(x) +- epsilon for all x currently
            # considered
            formula_1 = []

            if self.encoding == 'Real':
                # Use the encoding to calculate output for input x
                nn_y = self.predict(x)

                # Create variables for each output variable
                t_output_vars = [Real('y_{}_{}'.format(counter, i)) for i in range(len(self.nn_output_vars))]

                if counter > 0:
                    # Encode precision
                    for name, val in self.template.get_params().items():
                        prec_formula.append(Or(param_vars[name]-val>= precision, val-param_vars[name] >=precision))

            elif self.encoding == 'FP':
                epsilon = float(epsilon)
                # use NN to calculate output y for input x
                nn_y = self.nn_model.predict([x])[0]
                # create variables for each output value.
                t_output_vars = [FP(('y1_{}_{}'.format(counter, i)), Float32()) for i in range(len(self.nn_output_vars))]
            else:
                print('Encoding not supported.')
                sys.exit()

            # add encoding for function f according to template
            formula_1.append(self.template.generic_smt_encoding(x, t_output_vars))

            # add encoding to ensure output is within tolerance
            for i in range(len(self.nn_output_vars)):
                formula_1.append(cast(self.encoding, nn_y[i]) - t_output_vars[i] <= epsilon)
                formula_1.append(t_output_vars[i] - cast(self.encoding, nn_y[i]) <= epsilon)

            # Assert subformulas to solver.
            for sub in formula_1:
                solver_1.add(sub)

            # Check for satisfiability.
            print('Looking for new parameters')
            start_time_parameter = time.time()

            if self.encoding == 'Real' and counter > 0:
                solver_1.push()
                # Heuristic speed up
                for sub in prec_formula:
                    solver_1.add(sub)
                res = solver_1.check()

                # Try again without precision condition, if unsat
                if res == unsat:
                    solver_1.pop()
                    print('second check')
                    res = solver_1.check()
            else:
                res = solver_1.check()
            
            if res == sat:
                fo_model = solver_1.model()
                # Update parameters.
                new_params = {}

                for key in self.template.get_params():
                    var = self.template.param_variables()[key]
                    new_params[key] = value(self.encoding, fo_model, var)

                print('    -> New parameters found: ' + str(new_params))
                self.template.set_params(new_params)
            else:
                print('    -> Bound to strict: No parameters found.')
                return False, str(epsilon)

            end_time_parameter = time.time()
            print('    -> took', end_time_parameter-start_time_parameter, 'seconds')

            # 2nd condition:
            print('Looking for new input')
            if self.splits == 0:
                res, x = self._find_deviation(epsilon, refine=1)
            else:
                res, x = self._find_deviation_splitting(epsilon)
            
            counter += 1
        
        #Satisficing parameters found.
        return True, str(epsilon)

    def optimize_template(self, tolerance=0.001):
        """Function for optimizing parameters of a function-template
        to such that the maximal deviation is minimal.
        """

        if not self.encoding == 'Real':
            print('Optimiziation only possible using real-encoding.')
            return

        # Initial input
        x = self.lb
        print('Initial input: ', x)
        # Create a solver instance.
        solver_1 = Optimize()

        counter = 0
        epsilon = None

        # initialize the result with "satisfiable"
        res = sat

        # keep track of all absolute distances in each dimension
        all_abs = []

        # Define the maximal deviation
        max_epsilon = Real('max_epsilon')

        # while the encoding is satisfiable
        while res == sat:
            print('Optimizing parameters')
            start_time_opt = time.time()

            # Encode optimiziation problem, stored in formula_1:
            formula_1 = []

            # use NN to calculate output y for input x
            nn_y = self.predict(x)
            # create variables for each output value.
            t_output_vars = [Real('y1_{}_{}'.format(counter, i)) for i in range(len(self.nn_output_vars))]

            # add encoding for function f according to template
            formula_1.append(self.template.generic_smt_encoding(x, t_output_vars))

            # Firstly, encode the distance in each dimension
            absolutes = [Real('abs_{}_{}'.format(counter, i)) for i in range(len(self.nn_output_vars))]
            for i in range(len(self.nn_output_vars)):
                formula_1.append(absolutes[i] ==
                                 If(nn_y[i] - t_output_vars[i] >= 0,
                                    nn_y[i] - t_output_vars[i],
                                    t_output_vars[i] - nn_y[i]
                                    )
                                 )
            all_abs.extend(absolutes)

            # Secondly, sum up those distances
            maximze = And(
                Or([max_epsilon == abs for abs in all_abs]),
                And([max_epsilon >= abs for abs in all_abs])
            )
            if counter > 0:
                solver_1.pop()

            # Assert subformulas.
            for sub in formula_1:
                solver_1.add(sub)
            solver_1.push()
            solver_1.add(maximze)

            # Optimize parameters
            solver_1.minimize(max_epsilon)
            res = solver_1.check()
            if res == sat:
                fo_model = solver_1.model()
                # print(fo_model)

                # Update parameters.
                new_params = {}
                for key in self.template.get_params():
                    var = self.template.param_variables()[key]
                    new_params[key] = value(self.encoding, fo_model, var)
                
                print('    -> New parameters found: ' + str(new_params))
                self.template.set_params(new_params)

                epsilon = value(self.encoding, fo_model, max_epsilon)
                print('    -> Maximal distance:' + str(value('FP', fo_model, max_epsilon)) + '+-' + str(tolerance))
            else:
                print('Error! No satisfying parameters found.')
                print(res)
                return False, str(epsilon)

            counter += 1

            end_time_opt = time.time()
            print('    -> Took', end_time_opt-start_time_opt, 'seconds')

            # Encode 2nd condition:
            print('Looking for new input')
            res, x = self._find_deviation(epsilon+tolerance, refine=0)
            # print('End of while: ' + str(res)+str(x))
        
        return True, str(epsilon)

    def regression_verification_1d(self, size = 200, epsilon: float = 0.5, epsilon_accuracy_steps = 4):
        """Method for finding parameters of a function-template to fit the MLP with maximal deviation ``epsilon``.
        TODO: describe difference to optimize template

        Parameters:
            epsilon = 0.5: Tolerance of template. Within the domain ``interval`` (specified at construction time), the
            output of the closed form and the MLP are not allowed to differ more than ``epsilon``
        """

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
        end_time_regression = time.time()
        print('    -> Function found: f(x) = ', reg.coef_[0][0], 'x +', reg.intercept_[0])
        print('    -> took', end_time_regression - start_time_regression, 'seconds')
        print(self.template.get_params())

        # binary search for epsilon
        print('Calculating deviation range')
        lower = 0
        upper = epsilon

        #sanity check upper bound for binary search (epsilon)
        print('    -> Sanity check upper bound for binary search (epsilon)')
        if self.splits == 0:
            res, x = self._find_deviation(epsilon, refine=0)
        else:
            res, x = self._find_deviation_splitting(epsilon)
        if res == unsat:
            print('        * Passed: epsilon sufficiently large')
        else:
            print('        * Error: choose larger epsilon')
            return False, epsilon

        for _ in range(epsilon_accuracy_steps):
            print('Maximum deviation range: [', lower, ',', upper, ']')
            print('Searching for tighter bounds')
            mid = (lower + upper)/2
            if self.splits == 0:
                res, x = self._find_deviation(mid, refine=0)
            else:
                res, x = self._find_deviation_splitting(mid)
            # epsilon accuracy sufficient -> refine upper error bound (make it lower)
            if res == unsat:
                upper = mid
            # epsilon accuracy to tight tight -> refine lower error bound (make lower error bound larger)
            else:
                lower = mid

        print('Final maximum deviation range: [', lower, ',', upper, ']')
        print('For tighter bounds increase epsilon accuracy steps.')

        # Plot the results
        '''plt.scatter(x_samples, y_samples, c='deepskyblue')
        plt.plot(x_samples, reg.coef_[0][0] * x_samples + reg.intercept_[0], 'k')
        plt.show()
        plt.clf()'''

        return True, (lower, upper)

    def regression_verification_nd(self, func_class, sizes, epsilon: float = 0.5, epsilon_accuracy_steps=4):
        """Method for finding parameters of a function-template to fit the MLP with maximal deviation ``epsilon``.
        TODO: describe difference to optimize template

        Parameters:
            epsilon = 0.5: Tolerance of template. Within the domain ``interval`` (specified at construction time), the
            output of the closed form and the MLP are not allowed to differ more than ``epsilon``
        """

        # transform intervals into different format
        intervals = []
        for dimension in range(len(self.lb)):
            intervals += [[self.lb[dimension], self.ub[dimension]]]

        # create samples form input network
        print('Taking samples to do regression')

        # sanity check
        if len(intervals) != func_class.dimension():
            print('Error: dimension of', func_class.name(), 'is', func_class.dimension(), 'but you provided',
                  len(intervals), 'intervals')
        if len(sizes) != func_class.dimension():
            print('Error: dimension of', func_class.name(), 'is', func_class.dimension(), 'but you provided',
                  len(sizes), 'sizes')

        # use intervals and their sizes to create samples for each dimension separately
        interval_vectors = [np.linspace(intervals[dim][0], intervals[dim][1], sizes[dim]) for dim in
                            range(func_class.dimension())]

        # use the array of interval vectors to get an n-dimensional grid
        x_samples = np.vstack(np.meshgrid(*interval_vectors)).reshape(func_class.dimension(), -1).T

        # do predictions
        y_samples = self.nn_model.predict(x_samples)

        # do regression to find parameters
        print('Doing regression')
        start_time_regression = time.time()
        reg = LinearRegression().fit(x_samples, y_samples)

        # update template parameters
        new_params = {}
        for row_dim in range(func_class.dimension()):
            for col_dim in range(func_class.dimension()):
                new_params.update({'a{}{}'.format(row_dim+1,col_dim+1): reg.coef_[row_dim][col_dim]})
        for row_dim in range(func_class.dimension()):
            new_params.update({'b{}'.format(row_dim+1): reg.intercept_[row_dim]})
        self.template.set_params(new_params)
        end_time_regression = time.time()
        print('    -> Function found: f(x) = ')
        print(reg.coef_, 'x +', reg.intercept_)
        print('    -> took', end_time_regression - start_time_regression, 'seconds')

        # binary search for epsilon
        print('Calculating deviation range')
        lower = 0
        upper = epsilon

        # sanity check upper bound for binary search (epsilon)
        print('    -> Sanity check upper bound for binary search (epsilon)')
        if self.splits == 0:
            res, x = self._find_deviation(epsilon, refine=0)
        else:
            res, x = self._find_deviation_splitting(epsilon)
        if res == unsat:
            print('        * Passed: epsilon sufficiently large')
        else:
            print('        * Error: choose larger epsilon')
            return False, epsilon

        for _ in range(epsilon_accuracy_steps):
            print('Maximum deviation range: [', lower, ',', upper, ']')
            print('Searching for tighter bounds')
            mid = (lower + upper) / 2
            if self.splits == 0:
                res, x = self._find_deviation(mid, refine=0)
            else:
                res, x = self._find_deviation_splitting(mid)
            # epsilon accuracy sufficient -> refine upper error bound (make it lower)
            if res == unsat:
                upper = mid
            # epsilon accuracy to tight tight -> refine lower error bound (make lower error bound larger)
            else:
                lower = mid

        print('Final maximum deviation range: [', lower, ',', upper, ']')
        print('For tighter bounds increase epsilon accuracy steps.')

        return True, (lower, upper)

    def polyfit_verification_1d(self, func_class, size = 200, epsilon: float = 0.5, epsilon_accuracy_steps = 4):
        """Method for finding parameters of a function-template to fit the MLP with maximal deviation ``epsilon``.
        TODO: describe difference to optimize template

        Parameters:
            epsilon = 0.5: Tolerance of template. Within the domain ``interval`` (specified at construction time), the
            output of the closed form and the MLP are not allowed to differ more than ``epsilon``
        """

        # create samples form input network
        print('Taking samples to do Polynomial fit')
        x_samples = np.linspace(self.lb, self.ub, size)
        y_samples = self.nn_model.predict(x_samples)


        # do polyfit to find parameters
        print('Doing polynomial fit')

        # reshape from (n,1) to (n,)
        x_samples = x_samples.reshape((len(x_samples, )))
        y_samples = y_samples.reshape((len(y_samples, )))
        start_time_fit = time.time()
        fit = np.polynomial.polynomial.polyfit(x_samples, y_samples, func_class.degree())

        new_params = {}

        for degree in range(func_class.degree()+1):
            new_params.update({'a,{}'.format(degree): fit[degree]})

        self.template.set_params(new_params)

        end_time_fit = time.time()
        # print('    -> Function found: f(x) = ', reg.coef_[0][0], 'x +', reg.intercept_[0])
        # print('    -> took', end_time_fit - start_time_fit  , 'seconds')
        # print(self.template.get_params())

        # binary search for epsilon
        print('Calculating deviation range')
        lower = 0
        upper = epsilon

        #sanity check upper bound for binary search (epsilon)
        print('    -> Sanity check upper bound for binary search (epsilon)')
        if self.splits == 0:
            res, x = self._find_deviation(epsilon, refine=0)
        else:
            res, x = self._find_deviation_splitting(epsilon)
        if res == unsat:
            print('        * Passed: epsilon sufficiently large')
        else:
            print('        * Error: choose larger epsilon')
            return False, epsilon

        # this is only to find bugs
        testval = 0
        prediction = 0
        for degree in range(func_class.degree() + 1):
            prediction += fit[degree] * (testval ** degree)
        print('predicted value:', prediction)
        print('NN value', self.nn_model.predict([testval]))

        for _ in range(epsilon_accuracy_steps):
            print('Maximum deviation range: [', lower, ',', upper, ']')
            print('Searching for tighter bounds')
            mid = (lower + upper)/2
            if self.splits == 0:
                res, x = self._find_deviation(mid, refine=0)
            else:
                res, x = self._find_deviation_splitting(mid)
            # epsilon accuracy sufficient -> refine upper error bound (make it lower)
            if res == unsat:
                upper = mid
            # epsilon accuracy to tight tight -> refine lower error bound (make lower error bound larger)
            else:
                lower = mid

        print('Final maximum deviation range: [', lower, ',', upper, ']')
        print('For tighter bounds increase epsilon accuracy steps.')

        # calculate predictions for plotting
        y_predictions = []
        for i in range(len(x_samples)):
            y_prediction = 0
            for degree in range(func_class.degree()+1):
                y_prediction += fit[degree] * (x_samples[i]**degree)
            y_predictions += [y_prediction]
        y_predictions = np.array(y_predictions)

        # plot predictions
        '''plt.scatter(x_samples, y_samples, c='deepskyblue')
        plt.plot(x_samples, y_predictions, 'k')
        plt.show()
        plt.clf()'''
        return True, (lower, upper)


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
        print('*****************formula_2:')
        print(formula_2)

        # norm 1 distance TODO: combine with loop below
        deviation = Or(
                            Or(
                                [self.nn_output_vars[i] - self.template.output_variables()[i] > epsilon
                                    for i in range(len(self.nn_output_vars))]),
                            Or(
                                [self.template.output_variables()[i] - self.nn_output_vars[i] > epsilon
                                    for i in range(len(self.nn_output_vars))])
                        )
        print('*****************deviation:')
        print(deviation)

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
            fo_model = self.solver_2.model()

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
                if self.solver_2.check() == sat:
                    print('    -> improvement found')
                    fo_model = self.solver_2.model()
                else:
                    print('    -> no improvement found')
                    break

            # Extract new input for parameter correction.:
            x_list = [value(self.encoding, fo_model, var) for var in self.nn_input_vars]
            x = tuple(x_list)
            print('    -> New input found: ' + str(x))
            end_time_deviation = time.time()
            print('    -> took', end_time_deviation-start_time_deviation, 'seconds')
            return res, x
        else:
            end_time_deviation = time.time()
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
            print('    -> Most recent parameters correct within epsilon = +-', str(epsilon))
            return res, None

    def _init_worker_solvers(self):

        for _ in range(len(self.nn_model_formulas) - len(self.processes)):
            p = multiprocessing.Process(
                target=worker_solver,
                args=(
                    self.jobs,
                    self.solutions,
                    self.model_path,
                    str(self.template.__class__.__name__),
                    self.splits,
                    self.lb,
                    self.ub,
                    self.encoding
                    )
                )
            self.processes.append(p)
            p.start()

    def _find_deviation_splitting(self, epsilon):

        start_time_deviation = time.time()

        res, x = unsat, None

        # Number of workers
        workers_n = range(len(self.nn_model_formulas))
        self._init_worker_solvers()

        # Announce new jobs
        pickle_params = {key: pickleable_z3num(x) for key,x in self.template.get_params().items()}
        for i in workers_n:
            job = (i, epsilon, pickle_params)
            self.jobs.put(job)

        # Await solutions
        finished_workers = set()
        for _ in workers_n:
            worker_id, new_res, new_x = self.solutions.get()
            finished_workers.add(worker_id)
            if new_res == sat:
                # Save result
                res, x = new_res, tuple([reverse_pickleablility(x) for x in new_x])
                break

        # Kill unfinished processes
        live_processes = []
        for p in self.processes:
            if p.pid in finished_workers:
                live_processes.append(p)
            else:
                p.kill()

        self.processes = live_processes

        end_time_deviation = time.time()

        if res == sat:
            print('    -> New input found: ' + str(x))
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
        else:
            print('    -> took', end_time_deviation - start_time_deviation, 'seconds')
            print('Most recent parameters sufficient (epsilon = ' + str(epsilon) + ')')
            print()

            # Terminate processes
            for p in self.processes:
                p.kill()

        return res, x

    def test_encoding(self, input):
        """Function that tests whether solving the encoding for a
        MLP-model produces a correct model of the encoding.

        Parameters:
            input: tuple representing the input.
        """

        # Make a prediction
        prediction = self.predict(input)

        # Convert to readable decimal representation.
        res_dec = []
        for var in prediction:
            # This is a suspiciously hacky solution.
            res_dec.append(float(eval(str(var))))

        # Print the result for comparison.
        print('The calculated result is: ' + str(res_dec))
        print('However it should be:' + str(self.my_encoder.model.predict([input])))

    def predict(self, input):
        """Function that calculates a prediction of the NN based on its
        encoding.

        Parameters:
            input: tuple representing the input.
        """

        # Create whole encoding, if not yet done
        self.nn_model_formula, _, _ = self.my_encoder.encode()

        # Create a solver instance.
        solver = Solver()

        # Assert sub formulas.
        for v in self.nn_model_formula.values():
            solver.add(v)

        # Encode the input.
        input_formula = self.my_encoder.encode_input(input)
        for v in input_formula.values():
            solver.add(v)

        # Check for satisfiability.
        res = solver.check()
        if res != sat:
            print('ERROR. NN-Formula is not satisfiable.')
            sys.exit()
        fo_model = solver.model()

        # Convert to output list according to output vars.
        res_list = []
        for var in self.nn_output_vars:
            res_list.append(fo_model.eval(var, model_completion=True))

        return res_list

def value(encoding, fo_model, var):
    if encoding == 'Real':
        return fo_model.eval(var, model_completion=True)
    elif encoding == 'FP':
        return get_float(fo_model, var)
    else:
        print('Encoding not supported.')
        sys.exit()

def cast(encoding, var):
    if encoding == 'Real':
        return var
    elif encoding == 'FP':
        return float(var)
    else:
        print('Encoding not supported.')
        sys.exit()

def pickleable_z3num(val):
    if is_int_value(val):
        return (val.as_long(),)
    elif is_rational_value(val):
        return((val.numerator_as_long(),val.denominator_as_long()))
    else:
        print('Error. Value not rational.')

def reverse_pickleablility(val):
    if len(val)==1:
        return RealVal(val[0])
    elif len(val)==2:
        return RealVal(str(val[0])+'/'+str(val[1]))
    else:
        print('Error. Cannot reverse pickleability.')

def worker_solver(jobs, solutions,  model_path, template_name, splits, lb, ub, encoding):

    # initial preparation
    my_encoder = Encoder(model_path, encoding=encoding)
    nn_model_formulas, nn_output_vars, nn_input_vars = my_encoder.encode_splitted(number=splits)
    if str(template_name) == 'LinearTemplate':
        template = LinearTemplate(encoding=encoding)
    else:
        print('Template \"' + str(template_name) +'\" has to be added here.')
        sys.exit()

    # Input conditions.
    formula_2 = []
    for i in range(len(nn_input_vars)):
        formula_2.append(nn_input_vars[i] == template.input_variables()[i])
        formula_2.append(nn_input_vars[i] >= lb[i])
        formula_2.append(nn_input_vars[i] <= ub[i])

    while 1:
        # Block until new job is available
        formula_index, epsilon, new_params = jobs.get()
        print('Solving index: '+ str(formula_index) + ' and epsilon: '+ str(epsilon))

        # Update template parameters
        new_params = {key: reverse_pickleablility(x) for key,x in new_params.items()}
        template.set_params(new_params)
        print(new_params)
        print(template.get_params())

        # Encode the template.
        template_formula = template.smt_encoding()

        # distance
        deviation = Or(
                            Or(
                                [nn_output_vars[i] - template.output_variables()[i] > epsilon
                                    for i in range(len(nn_output_vars))]),
                            Or(
                                [template.output_variables()[i] - nn_output_vars[i] > epsilon
                                    for i in range(len(nn_output_vars))])
                        )

        total_split = formula_2 + nn_model_formulas[formula_index] + [deviation, template_formula]

        # Solve the formula
        solver = Solver()
        for subformula in total_split:
            solver.add(subformula)
        res = solver.check()

        # Extract the model, if sat
        x = None
        if res == sat:
            fo_model = solver.model()
            x_list = [value(encoding, fo_model, var) for var in nn_input_vars]
            x = [pickleable_z3num(x) for x in x_list]

        solutions.put((getpid(), res, x))
