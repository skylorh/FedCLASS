from models.icarl import ICARL
from models.ours import OURS
from models.glfc import GLFC
from models.ce import CE
from models.wa import WA

from utils.fed_algorithm import FedAvg

def get_model(args, data_manager):
    name = args["model_name"].lower()
    if name == "icarl":
        return ICARL(args, data_manager)
    elif name == "ours":
        return OURS(args, data_manager)
    elif name == "glfc":
        return GLFC(args, data_manager)
    elif name == "ce":
        return CE(args, data_manager)
    elif name == "wa":
        return WA(args, data_manager)
    else:
        assert 0
        
def get_fed_algo(args, train_sizes, init_model, lr):
    if args["fed_algo"] == "fedavg":
        return FedAvg(train_sizes, init_model)
    else:
        assert 0