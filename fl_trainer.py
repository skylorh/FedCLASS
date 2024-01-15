from select import select
import logging
import copy
from utils import factory
from utils.data_manager import DataManager
from utils.wandb import WandbUploader
from utils.trainer_utils import *
import numpy as np
import json
import argparse


class FlTrainer(object):
    def __init__(self, args):
        
        usewdb = args["usewdb"]
        self.wdbup = None
        if usewdb:
            self.wdbup = WandbUploader(args, "")
        
        self.model_name = args["model_name"].lower()
        save_old_network_list = ["ours", "glfc", "wa"]
        self.save_old_network_on = self.model_name in save_old_network_list
        
        set_logs(args)
        set_random(args)
        set_device(args)
        set_print(args)
        print_args(args)
        
        self.data_manager = DataManager(args)
    

    def fl_train(self, args):

        global_model = factory.get_model(args, self.data_manager)
        print(global_model.network)
        
        all_ids = np.arange(args["total_clients"], dtype=int)
        clients = np.array([factory.get_model(args, self.data_manager) for i in range(args["total_clients"])])

        for task_id in range(self.data_manager.nb_tasks):

            global_model.total_classes_old_global = copy.deepcopy(global_model.total_classes_global)
            global_model.set_info_task_global(task_id)
            
            train_sizes = []
            for id in all_ids:
                cur_classes = clients[id].gen_cur_classes(task_id)
                clients[id].set_info_task_global(task_id)
                if args["memory_size"] > 0:
                    clients[id].set_train_loader_with_memory()
                else:
                    clients[id].set_train_loader_without_memory()
                
                logging.info("Client{} train_dataset len: {}".format(id+1, clients[id].train_dataset.__len__()))
                train_sizes.append(clients[id].train_dataset.__len__())

            for round_id in range(args["aggregation_round_per_task"]):

                logging.info(
                    "\n\nT{} R{}\n".format(task_id, round_id)
                )

                global_model.set_info_task_cur(task_id)
                global_model.set_info_round_cur(round_id)
                
                if round_id == 0:
                    global_model.update_network_fc()
                    
                    global_model.set_test_loader_global()
                
                select_ids = np.random.choice(args["total_clients"], args["select_clientnum"], replace=False)
                unselect_ids = np.setdiff1d(all_ids, select_ids)
                logging.info(
                    "Select: {} Unselect:{}".format(select_ids+1, unselect_ids+1)
                )
                for client in clients:
                    client.set_info_task_cur(task_id)
                    client.set_info_round_cur(round_id)

                logging.info(
                    "\nGen and load personal global model to selected clients...\n"
                )
                for id in select_ids:
                    clients[id].network = copy.deepcopy(global_model.network)
                        
                logging.info(
                    "\nSelected clients start training...\n"
                )
                for id in select_ids:
                    prev_key = "C{}-T{}-R{}-".format(id+1, task_id, round_id)
                    clients[id].train()
                    
                logging.info(
                    "\nSaving models and eval...\n"
                )
                fed_algo = factory.get_fed_algo(args, train_sizes, global_model.network, args["lr"])
                local_models = [clients[id].network for id in select_ids]
                client_indices = [id for id in select_ids]
                update_model = fed_algo.update(local_models, client_indices, global_model.network)
                global_model.network.load_state_dict(update_model)
                
                test_acc_global_classes_top1, test_acc_global_classes_top3, old_task_acc, acc_map_global, acc_by_task = global_model.test("AfterAggregation")
                
                if self.wdbup!= None:
                    log_dict = {}
                    log_dict["test_acc_global_classes_top1"] = test_acc_global_classes_top1
                    log_dict["test_acc_global_classes_top3"] = test_acc_global_classes_top3
                    log_dict["old_task_acc"] = old_task_acc
                    
                    for i in range(self.data_manager.nb_tasks):
                        acc = 0.0
                        if i <= global_model.cur_task:
                            acc = acc_by_task[i]
                        log_dict["acc_task_" + str(i)] = acc

                    logging.info(log_dict)
                    self.wdbup.uploader.log(log_dict)
                
                if round_id == args["aggregation_round_per_task"] - 1:
                    logging.info(
                        "\nEnd last round of T{}\n".format(task_id)
                    )
                    if task_id != self.data_manager.nb_tasks - 1:
                        for client in clients:
                            client.build_rehearsal_memory(client.samples_per_class)
                            if self.save_old_network_on:
                                client.save_old_network(global_model.network)
                                client.total_classes_old_global = copy.deepcopy(global_model.total_classes_global)
                            
def main():
    args = setup_parser().parse_args()
    with open(args.config) as param_file:
        param = json.load(param_file)
    args = vars(args)  
    args.update(param)  
    print(args)
    
    flTrainer = FlTrainer(args)
    flTrainer.fl_train(args)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./exps/ce.json',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()

