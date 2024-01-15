import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.cuda.amp import GradScaler, autocast
from utils.inc_net import GLFCNet

class GLFC(BaseLearner):
    def __init__(self, args, data_manager):
        super().__init__(args, data_manager)
        
        self.network = GLFCNet(args["convnet_type"], False)

    def train(self):

        logging.info(
            "T{}-R{}-C{} Learning on classes:{}".format(self.cur_task, self.cur_round, self.client_id, self.cur_classes)
        )
        logging.info("self.total_classes_global: {}".format(self.total_classes_global))
        logging.info("self.total_classes_local: {}".format(self.total_classes_local))
        
        self._train()
        
    def _train(self):
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

                    targets_one_hot = self.get_one_hot(targets, self.cur_end, self.device)
                    
                    logits, targets_one_hot = logits.cuda(self.device), targets_one_hot.cuda(self.device)
                    
                    if self.old_network == None:
                        w = self.efficient_old_class_weight(logits, targets)
                        loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none'))

                        loss = loss_cur
                    else:
                        w = self.efficient_old_class_weight(logits, targets)
                        loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none'))

                        distill_target = targets_one_hot.clone()
                        
                        old_target = torch.sigmoid(self.old_network(inputs))
                        old_task_size = old_target.shape[1]
                        distill_target[..., :old_task_size] = old_target
                        loss_old = F.binary_cross_entropy_with_logits(logits, distill_target)

                        loss = 0.5 * loss_cur + 0.5 * loss_old

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


    def get_one_hot(self, target, num_class, device):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
        one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot

    def efficient_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        
        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = self.get_one_hot(label, self.cur_end, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if self.cur_start != 0:
            for i in range(self.cur_start):
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))
 
            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)

            w = w1 + w2
        
        else:
            w = g.clone().fill_(1.)

        return w

    




