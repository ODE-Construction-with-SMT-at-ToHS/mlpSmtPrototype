import os
import argparse


def _is_valid_file(arg):
    """
    Type check for optional argument '--model'. A path to an .h5 Keras model file should be passed.
    """

    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('{} not found!'.format(arg))
    elif not (os.path.splitext(arg)[1] == ".h5"):
        raise argparse.ArgumentTypeError('{} is not a valid .h5 Keras model file!'.format(arg))
    else:
        return arg


def parse_args():
    """
    Specifies valid arguments
    """

    # default path to model
    path_to_model = os.path.join('mlp_smt_closed', 'mlp', 'models', 'LinearA' + '_model.h5')

    # init parser
    parser = argparse.ArgumentParser(description="DESCRIPTION", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optional Arguments
    parser.add_argument('-m', '--model', default=path_to_model, type=_is_valid_file,
                        help='path to Keras .h5 model file')
    parser.add_argument('--method', default='smt', choices=['smt', 'ls'],
                        help='to find parameters choose either the SMT (smt) method or the least-squares (ls) method')
    parser.add_argument('-o', '--optimize', action='store_true',
                        help='chose whether the parameters should not only be found but also optimized. Only relevant '
                             'when also choosing "--method smt"')
    parser.add_argument('-s', '--splits', default=0, type=int,
                        help='#splits + 1 threads are used')
    parser.add_argument('-t', '--template', default='linear', choices=['linear', 'polynomial'],
                        help='choose to fit the MLP to a linear or polynomial function')
    parser.add_argument('-d', '--degree', default=1, type=int, help='degree of the polynomial')
    parser.add_argument('-i', '--intervals', default=((-8,), (8,)),
                        help='interval in which the parameters should fit')
    parser.add_argument('--sizes', default=[200],
                        help='number of samples taken in case "--method ls"')
    parser.add_argument('-e', '--epsilon', default=0.05, type=float,
                        help='Attention: the interpretation of epsilon is context dependent! Read the README or the '
                             'Documentation for details.')
    parser.add_argument('--steps', default=4, type=int,
                        help='number of interval refinement steps taken in case "--method ls"')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot fit if supported for this dimension' )

    args = parser.parse_args()

    return args
