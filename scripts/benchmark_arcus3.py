"""Disposable two-update preflight through exactly the dense training path."""
from train_arcus3 import main,parser
if __name__=='__main__':
    args=parser().parse_args();args.preflight=True;main(args)
