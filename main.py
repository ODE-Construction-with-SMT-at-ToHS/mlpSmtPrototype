from encoder import *
import arguments

if __name__ == '__main__':
    
    #TODO Arguments are ignored for now
    args = arguments.parse_args()
    
    model_path = args.model
    myEncoder = Encoder(model_path)
    model_formula = myEncoder.encode(x=1)

    print(model_formula)
