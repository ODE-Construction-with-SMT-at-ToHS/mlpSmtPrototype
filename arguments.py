import os
import argparse

def _is_valid_file(arg):

    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('{} not found!'.format(arg))
    elif not os.path.splitext(arg)[1] == ".pddl":
        raise argparse.ArgumentTypeError('{} is not a valid .h5 Keras model file!'.format(arg))
    else:
        return arg

def parse_args():
    """
    Specifies valid arguments
    """

    parser = argparse.ArgumentParser(description= "DESCRIPTION", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optional Arguments

    parser.add_argument('-model', help='Path to Keras .h5 model file', default='models\Linear2DA_model.h5')

    parser.add_argument('-template', help='Specification of which template is to be configured') #possibly enum type?

    args = parser.parse_args()

    return args
