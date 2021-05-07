import argparse

def parse_args():
    """
    Specifies valid arguments
    """

    parser = argparse.ArgumentParser(description= DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter())

    # Optional Arguments

    parser.add_argument('-model', help='Path do model file')

    parser.add_argument('-template', help='Specification of which template is to be configured') #possibly enum type?

    args = None

    return args
