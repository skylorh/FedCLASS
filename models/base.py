import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy

class BaseLearner(object):
    count_id = 0
    def __init__(self, args, data_manager):
        self.client_id = BaseLearner.count_id
        BaseLearner.count_id += 1
        
        self.args = args
        self.data_manager = data_manager
        
        self.nb_tasks = data_manager.nb_tasks
        self.cur_task = -1
        self.cur_round = -1
        self.cur_start = 0
        self.cur_end = 0
        self.network = None
        self.old_network = None
        self.optimizer = None
        
        self.data_memory, self.targets_memory = np.array([]), np.array([])
        self.total_classes_local_known = np.array([], dtype=np.int)
        self.total_classes_local = np.array([], dtype=np.int)
        self.total_classes_global = np.array([], dtype=np.int)
        self.total_classes_old_global = np.array([], dtype=np.int)
        
        self.cur_classes = np.array([], dtype=np.int)
        self.total_classes_cur_task = np.array([], dtype=np.int)
        
        # 通用参数
        self.increment = args["increment"]
        self.init_cls = args["init_cls"]

        self.epoches = args["epoches"]
        self.lr = args["lr"]
        self.weight_decay = args["weight_decay"]
        self.momentum = args["momentum"]

        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        
        self.step_size = args["increment"]
        
        self.local_classnum_per_task_init = args["local_classnum_per_task_init"]
        self.local_classnum_per_task = args["local_classnum_per_task"]
        self.memory_size = args["memory_size"]
        self.memory_size_per_class = args["memory_size_per_class"]
        self.fixed_memory = args["fixed_memory"]
        self.device = args["device"][0]
        
    def set_info_task_cur(self, task_id):
        self.cur_task = task_id
        self.cur_start = self.data_manager._classes_select_increments[task_id][0]
        self.cur_end = self.data_manager._classes_select_increments[task_id][1]
        self.total_classes_cur_limitcurtask = np.arange(self.data_manager._classes_select_increments[task_id][0], self.data_manager._classes_select_increments[task_id][1], dtype=int)
        
    def set_info_round_cur(self, round_id):
        self.cur_round = round_id
        
    def set_info_task_global(self, task_id):
        self.total_classes_global = np.arange(self.data_manager._classes_select_increments[task_id][1], dtype=int)
        self.total_classes_global_limitcurtask = np.arange(self.data_manager._classes_select_increments[task_id][0], self.data_manager._classes_select_increments[task_id][1], dtype=int)
 
    def update_network_fc(self):
        self.network.update_fc(self.total_classes_global.__len__())
    
    def set_train_loader_with_memory(self):
        self.train_dataset = self.data_manager.get_dataset(
            self.cur_classes,
            source="train",
            mode="train",
            appendent=self.get_memory(),
            client_id = self.client_id # self.client_id
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def set_train_loader_without_memory(self):
        self.train_dataset = self.data_manager.get_dataset(
            self.cur_classes,
            source="train",
            mode="train",
            client_id = self.client_id # self.client_id
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def set_test_loader_local(self):
        self.test_dataset_local = self.data_manager.get_dataset(
            self.total_classes_local,
            source="test",
            mode="test"
        )
        self.test_loader_local = DataLoader(
            self.test_dataset_local,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def set_test_loader_global(self):
        self.test_dataset_global = self.data_manager.get_dataset(
            self.total_classes_global,
            source="test",
            mode="test"
        )
        self.test_loader_global = DataLoader(
            self.test_dataset_global,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    @property
    def samples_per_class(self):
        if self.fixed_memory:
            return self.memory_size_per_class
        else:
            assert self.total_classes_local.__len__() != 0, "Total classes is 0"
            return self.memory_size // self.total_classes_local.__len__()

    def build_rehearsal_memory(self, per_class):
        if self.fixed_memory:
            self._construct_exemplar_unified(per_class)
        else:
            self._reduce_exemplar(per_class)
            self._construct_exemplar(per_class)
        self.total_classes_local_known = np.union1d(self.total_classes_local_known, self.cur_classes)

    def get_memory(self):
        logging.info("Client{} len(self.data_memory): {}".format(self.client_id, len(self.data_memory)))
        if len(self.data_memory) == 0:
            return None
        else:
            return (self.data_memory, self.targets_memory)


    def _reduce_exemplar(self, m):
        logging.info("Client{} Reducing exemplars...(max {} per classes)".format(self.client_id, m))
        dummy_data, dummy_targets = copy.deepcopy(self.data_memory), copy.deepcopy(self.targets_memory)
        self.data_memory, self.targets_memory = np.array([]), np.array([])

        for class_idx in self.total_classes_local_known.tolist():
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask], dummy_targets[mask]
            if len(dd) > m :
                dd, dt = dd[:m], dt[:m]
            self.data_memory = (
                np.concatenate((self.data_memory, dd))
                if len(self.data_memory) != 0
                else dd
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, dt))
                if len(self.targets_memory) != 0
                else dt
            )

    def _construct_exemplar(self, m):
        logging.info("Client{} Constructing exemplars...(max {} per classes)".format(self.client_id, m))
        for class_idx in self.cur_classes.tolist():
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                client_id = self.client_id # self.client_id
            )
            selected_exemplars = None
            exemplar_targets = None
            if len(data) > m:
                selected_exemplars = np.array(data[:m])
                exemplar_targets = np.full(m, class_idx)
            else:
                selected_exemplars = np.array(data)
                exemplar_targets = np.full(len(data), class_idx)
                
            self.data_memory = (
                np.concatenate((self.data_memory, selected_exemplars))
                if len(self.data_memory) != 0
                else selected_exemplars
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, exemplar_targets))
                if len(self.targets_memory) != 0
                else exemplar_targets
            )
    
    def _construct_exemplar_unified(self, m):
        logging.info("Client{} Constructing exemplars...(max {} per classes)".format(self.client_id, m))
        for class_idx in self.cur_classes.tolist():
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                client_id = self.client_id # self.client_id
            )
            selected_exemplars = None
            exemplar_targets = None
            if len(data) > m:
                selected_exemplars = np.array(data[:m])
                exemplar_targets = np.full(m, class_idx)
            else:
                selected_exemplars = np.array(data)
                exemplar_targets = np.full(len(data), class_idx)
                
            self.data_memory = (
                np.concatenate((self.data_memory, selected_exemplars))
                if len(self.data_memory) != 0
                else selected_exemplars
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, exemplar_targets))
                if len(self.targets_memory) != 0
                else exemplar_targets
            )
    
    def compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    
    def compute_accuracy_by_class(self, model, loader):
        model.eval()
        correct, total = 0, 0
        correct_top3 = 0

        correct_list = np.zeros(self.total_classes_global.__len__(), dtype=int)
        total_list = np.zeros(self.total_classes_global.__len__(), dtype=int)
        acc_map = {}

        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                logits = model(inputs)

            predicts = torch.max(logits, dim=1)[1]
            predicts = predicts.cpu()

            ### top1 ###
            for idx in range(targets.shape[0]):
                if predicts[idx] == targets[idx]:
                    correct_list[targets[idx].item()] += 1
                total_list[targets[idx].item()] += 1

            correct += (predicts == targets).sum()
            total += len(targets)
            
            ### top3 ###
            _, predicts_top3 = logits.topk(3, 1, True, True)
            predicts_top3 = predicts_top3.cpu()
            y_resize = targets.view(-1,1)
            correct_top3 += torch.eq(predicts_top3, y_resize).sum().float().item()
        
        acc_top1 = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        acc_top3 = correct_top3 * 100 / total
        
        print("acc_top1: {}, acc_top3: {}".format(acc_top1, acc_top3))
        
        for i in range(0, self.total_classes_global.__len__()):
            if total_list[i] != 0:
                acc_map[self.total_classes_global[i]] = correct_list[i] / total_list[i]
            else:
                acc_map[self.total_classes_global[i]] = 0
        
        
        return acc_top1, acc_top3, acc_map
    
    def gen_cur_classes(self, task_id):
        
        num = self.local_classnum_per_task_init
        if task_id != 0:
            num = self.local_classnum_per_task
        
        self.cur_classes = np.sort(
            np.random.choice(
                range(
                    self.data_manager._classes_select_increments[task_id][0],
                    self.data_manager._classes_select_increments[task_id][1]
                ),
                num,
                replace=False
            )
        )
        self.total_classes_cur_task = np.arange(
            self.data_manager._classes_select_increments[task_id][0],
            self.data_manager._classes_select_increments[task_id][1]
        )
        self.total_classes_local = np.union1d(self.total_classes_local, self.cur_classes)
        
        return copy.deepcopy(self.cur_classes)
        
    def test(self, info):
        self.network.to(self.device)
        self.network.eval()
        
        test_acc_global_classes_top1, test_acc_global_classes_top3, acc_map_global = self.compute_accuracy_by_class(self.network, self.test_loader_global)
        
        acc_by_task = {}
        old_task_acc = 0.0
        for i in range(self.cur_task+1):
            pairs = self.data_manager._classes_select_increments[i]
            acc = 0.0
            for j in range(pairs[0], pairs[1]):
                acc += acc_map_global[j]
            acc /= (pairs[1] - pairs[0])
            if i < self.cur_task:
                old_task_acc += acc
            acc_by_task[int(i)] = acc
            
        if self.cur_task > 0:
            old_task_acc /= self.cur_task
        
        acc_by_class = {}
        for i in self.total_classes_global:
            acc_by_class[int(i)] = acc_map_global[i]

        return test_acc_global_classes_top1, test_acc_global_classes_top3, old_task_acc, acc_map_global, acc_by_task
    
    def save_old_network(self, old_network):
        self.old_network = copy.deepcopy(old_network)
        for param in self.old_network.parameters():
            param.requires_grad = False


    