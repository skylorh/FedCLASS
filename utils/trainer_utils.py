from select import select
import sys
import logging
import torch
import os
import numpy as np
import random


def set_device(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device_type = args["device"]
    gpus = []
    gpu_ids = []

    for device_id in device_type:
        if device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device_id))
        gpus.append(device)
        gpu_ids.append(device_id)
        
    args["device"] = gpus

def set_random(args):
    seed = args["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_print(args):
    torch.set_printoptions(profile="full")
    
def set_logs(args):
    logs_name = "logs/{}/".format(args["model_name"])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            args["model_name"],
            args["prefix"],
            args["seed"],
            args["model_name"],
            args["convnet_type"],
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["total_clients"],
            args["select_clientnum"],
            args["local_classnum_per_task_init"],
            args["local_classnum_per_task"],
            args["epoches"],
            args["aggregation_round_per_task"],
            args["dirichlet_alpha"],
            args["dataset_partition"],
            args["fed_algo"],
            args["memory_size"]
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s][%(funcName)s][%(lineno)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def print_args(args):
    logging.info("pid: {}".format(os.getpid()))
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
