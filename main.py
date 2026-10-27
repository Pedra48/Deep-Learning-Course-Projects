from src.model import Model
from src.exp import Exp
args = ""
def main():
    model = Model(args)
    exp = Exp(args)
    exp.train()
    exp.eval()
    exp.test()
    

if __name__ == "__main__":
    main()