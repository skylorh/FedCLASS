import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.cuda.amp import GradScaler, autocast
from utils.inc_net import WANet

class WA(BaseLearner):
    def __init__(self, args, data_manager):
        super().__init__(args, data_manager)
        
        self.network = WANet(args["convnet_type"], False)
        
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
            
            for i, (indices, inputs, targets) in enumerate(self.train_loader):
                
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
        
        kd_lambda = self.total_classes_old_global.__len__() / self.total_classes_global.__len__()
        T = 2
        
        for epoch in range(self.epoches):
        
            losses = 0.0
            correct, total = 0, 0
            
            for i, (indices, inputs, targets) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                
                with autocast(enabled=True, dtype=torch.float16):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    logits = self.network(inputs)
                    
                    loss_clf = F.cross_entropy(logits, targets)
                    
                    
                    loss_kd = _KD_loss(
                        logits[:, : self.total_classes_old_global.__len__()],
                        self.old_network(inputs),
                        T,
                    )
                    
                    loss = (1-kd_lambda) * loss_clf + kd_lambda * loss_kd
                    
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
    
                
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



