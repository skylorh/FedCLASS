import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.cuda.amp import GradScaler, autocast
from utils.inc_net import OURSNet

class OURS(BaseLearner):
    def __init__(self, args, data_manager):
        super().__init__(args, data_manager)
        
        self.network = OURSNet(args["convnet_type"], False)
        
        self.temperature_old = args["temperature_old"]
        self.temperature_new = args["temperature_new"]
        
        self.alpha = args["alpha"]
        

    def train(self):

        logging.info(
            "T{}-R{}-C{} Learning on classes:{}".format(self.cur_task, self.cur_round, self.client_id, self.cur_classes)
        )
        logging.info("self.total_classes_global: {}".format(self.total_classes_global))
        logging.info("self.total_classes_local: {}".format(self.total_classes_local))
        
        if self.cur_task == 0:
            self._init_train()
        else:
            self._train()
        
    def _init_train(self):
        self.network.to(self.device)
        self.network.train()
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            momentum=self.momentum,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        scaler = GradScaler()
        
        for epoch in range(self.epoches):
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                
                with autocast(enabled=True, dtype=torch.float16):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    logits = self.network(inputs)
                    loss = F.cross_entropy(logits, targets)
                    
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)


            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Round{}, Client{}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                self.cur_round,
                self.client_id,
                epoch + 1,
                self.epoches,
                losses / len(self.train_loader),
                train_acc,
            )
            logging.info(info)
        
    def _train(self):
        self.network.to(self.device)
        self.network.train()
        self.old_network.eval()
        
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            momentum=self.momentum,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        scaler = GradScaler()
        
        for epoch in range(self.epoches):
        
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                
                with autocast(enabled=True, dtype=torch.float16):
                        
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    logits = self.network(inputs)
                    old_logits = self.old_network(inputs)
                    
                    loss_il = F.cross_entropy(logits, targets)
                    
                    softmax_logits = F.softmax(logits / self.temperature_new, dim=1)
                    
                    softmax_old_logits = F.softmax(old_logits / self.temperature_old, dim=1)
                    
                    
                    Z = F.softmax(logits / self.temperature_old, dim=1).clone()
        
                    dim0, dim1 = Z.shape
                    for i in range(dim0):
                        ratio = 1
                        ratio -= torch.sum(softmax_logits[i][self.cur_start: self.cur_end].clone())
                        Z[i][:self.cur_start] = softmax_old_logits[i][:self.cur_start].clone() * ratio
                    
                    loss_kd = 0.0
                    for i in range(dim0):
                        KL = 0.0
                        for j in range(dim1):
                            KL += softmax_logits[i][j] * torch.log(torch.clip(1e-8 + softmax_logits[i][j] / (1e-9 + Z[i][j].clone()), 1e-9, 1e3))
                        loss_kd += KL
                    loss_kd *= self.alpha / dim0
                        
                    loss = loss_il + loss_kd
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)


            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Round{}, Client{}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                self.cur_round,
                self.client_id,
                epoch + 1,
                self.epoches,
                losses / len(self.train_loader),
                train_acc,
            )
            logging.info(info)
    
    
    