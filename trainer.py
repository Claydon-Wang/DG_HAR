from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytz
from tensorboardX import SummaryWriter

import tqdm
import socket

criterion = nn.CrossEntropyLoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, args, model, train_loader, val_loader, optim):
        self.args = args
        self.model = model
        self.optim = optim
        self.lr = args.lr
        self.lr_decrease_rate = args.lr_decrease_rate
        self.batch_size = args.batch_size
        self.interval_validate=args.interval_validate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))
        self.out = args.out

        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = 0
        self.iteration = 0
        self.best_acc = 0.0
        self.best_epoch = -1

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (image, target, domain) in enumerate(self.val_loader):

                image = image.cuda()   #(B, 1, 128, 9)
                target = target.cuda() # 6 class
                domain = domain.cuda() # 5 domain

                output = self.model(image)
                val_loss += criterion(output, target).sum()

                prediction = output.data.max(1)[1]
                total_correct += prediction.eq(target.data.view_as(prediction)).sum()
                total += prediction.shape[0]


        val_loss /= total
        acc = float(total_correct) / total
        record_str = '\nTest Avg. Loss: %f, Accuracy: %f' % (val_loss.data.item(), acc)
        print(record_str)

        # writing log
        with open(osp.join(self.out, 'log.txt'), 'a') as f:
            f.write(record_str)

        # save model or not    
        if self.args.save_model == True:
            if self.epoch + 1 == self.args.n_epoch:
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, osp.join(self.args.out, '%s_%s_checkpoint.pth' % (self.args.dataset, self.args.target_domain) ))


    def train_epoch(self):
        self.model.train()
        self.running_cls_loss = 0.0
        start_time = timeit.default_timer()

        # for source_loader in self.train_loader:
        for batch_idx, (image, target, domain) in enumerate(self.train_loader):

            image = image.cuda()   #(B, 1, 128, 9)
            target = target.cuda() # 6 class
            domain = domain.cuda() # 5 domain
            # print(domain)


            assert self.model.training
            self.optim.zero_grad()

            output = self.model(image)

            loss = criterion(output, target)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            self.running_cls_loss += loss.item()

            loss.backward()
            self.optim.step()


        self.running_cls_loss /= len(self.train_loader)
        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average clsLoss: %f, Execution time: %.5f' %
              (self.epoch, get_lr(self.optim), self.running_cls_loss, stop_time - start_time))
        




    def train(self):
        self.model.train()
        for epoch in range(self.args.n_epoch):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            # adjust learning rate
            if (self.epoch + 1) % self.args.lr_decrease_interval == 0:
                _lr_gen = self.lr * self.lr_decrease_rate
                self.lr = _lr_gen
                # print(self.lr)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = _lr_gen

            # validate model
            if (self.epoch + 1) % self.interval_validate == 0:
                self.validate()