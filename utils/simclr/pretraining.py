# Based on https://github.com/HobbitLong/SupContrast

from __future__ import print_function

import workspace
from datasets.data import get_pretraining_dataset, load_dataset
from models.models import get_encoder

import torch.nn as nn
import torch.nn.functional as F

import logging

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from utils.simclr.util import TwoCropTransform, AverageMeter
from utils.simclr.util import adjust_learning_rate, warmup_learning_rate
from utils.simclr.util import set_optimizer, save_model
# from networks.resnet_big import SupConResNet
from utils.simclr.losses import SupConLoss
import utils.utils as utils

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


log = logging.getLogger(__name__)


def add_option(opt):

    opt.model_path = os.path.join(opt.save_dir, f"{opt.data.dataset}_models")
    opt.tb_path = os.path.join(opt.save_dir, f"{opt.data.dataset}_tensorboard")

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.data.dataset, opt.model.base_model_type, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    ds = load_dataset(opt.data.dataset, opt.noise_type)
    train_dataset = get_pretraining_dataset(opt.data.dataset, ds)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader



class SupConModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, encoding_size, feat_dim=128):
        super(SupConModel, self).__init__()
        self.encoder = model
        self.head = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(inplace=True),
            nn.Linear(encoding_size, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


def set_model(opt):

    model, encoding_size = get_encoder(opt, load_pretrained=False)
    model = SupConModel(model, encoding_size)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (_, images, _, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        bsz = images.shape[0] // 2

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            raise NotImplementedError("Hasn't been implemented for l_labels")
            # loss = criterion(features, l_labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            log.info('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main(config):

    config = add_option(config)

    # Get the device and set the seed
    device = utils.set_device(config)
    if config.seed is not None:
        utils.set_seed(config.seed)

    # build data loader
    train_loader = set_loader(config)

    # build model and criterion
    model, criterion = set_model(config)

    # build optimizer
    optimizer = set_optimizer(config, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=config.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, config.epochs + 1):
        adjust_learning_rate(config, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, config)
        time2 = time.time()
        log.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % config.save_freq == 0:
            save_file = os.path.join(
                config.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, config, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        config.save_folder, 'last.pth')
    save_model(model, optimizer, config, config.epochs, save_file)



if __name__ == "__main__":

    # Load the config and create the results folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_name = sys.argv[1]
    config = workspace.create_workspace(current_dir, config_file_name)

    # Run the experiment
    main(config)
