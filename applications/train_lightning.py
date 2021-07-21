# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time
import torch
import poptorch
import logging
import popdist
import horovod.torch as hvd

from poptorch.optim import SGD, RMSprop, AdamW

import pytorch_lightning as pl
from pytorch_lightning.plugins import IPUPlugin

import sys
import os


if not os.path.exists('examples/applications/pytorch/cnns/train'):
    print("Could not find the CNN examples directory, have you run git submodule init/update?")
    exit()

sys.path.append('examples/applications/pytorch/cnns/')
sys.path.append('examples/applications/pytorch/cnns/train')

from train_utils import parse_arguments
from validate import create_validation_opts
import models
import utils
import datasets

from lr_schedule import WarmUpLRDecorator, PeriodicLRDecorator
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


class LightningResnet(pl.LightningModule):
    def __init__(self, model, label_smoothing=0.0, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.model = model
        self.label_smoothing = 1.0 - label_smoothing
        self.loss = torch.nn.NLLLoss(reduction="mean")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch, _):
        input, label = batch

        # Calculate loss in full precision
        output = self.model(input).float()
        log_preds = torch.nn.functional.log_softmax(output, dim=1)

        loss_items = {}
        loss_items['classification_loss'] = self.label_smoothing * self.loss(log_preds, label)
        if self.label_smoothing > 0.0:
            # cross entropy between uniform distribution and output distribution
            loss_items["smoothing_loss"] = - torch.mean(log_preds) * self.label_smoothing
        else:
            loss_items["smoothing_loss"] = torch.zeros(1)

        final_loss = loss_items["smoothing_loss"] + loss_items["classification_loss"]
        return poptorch.identity_loss(final_loss, reduction='mean')

    def validation_step(self, batch, _):
        input, labels = batch
        output = self.model(input).float()
        return utils.accuracy(output, labels)

    # Print the validation accuracy only on an epoch level.
    def validation_epoch_end(self, outputs) -> None:
        self.log('validation_accuracy', torch.stack(outputs).mean(), prog_bar=True)

    def forward(self, input, hidden):
        return self.network(input, hidden)

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]


def create_model_opts(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options(ipus_per_replica=len(opts.pipeline_splits) + 1)
    else:
        model_opts = poptorch.Options()
        model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iterations)
    # Set mean reduction
    model_opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    if opts.seed is not None:
        model_opts.randomSeed(opts.seed)
    return model_opts


def get_optimizer(opts, model):
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {'params': regularized_params, 'weight_decay': opts.weight_decay},
        {'params': non_regularized_params, 'weight_decay': 0}
    ]

    optimizer = None
    if opts.optimizer == 'sgd':
        optimizer = SGD(params, lr=opts.lr, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling)
    elif opts.optimizer == 'sgd_combined':
        optimizer = SGD(params, lr=opts.lr, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling, velocity_scaling=opts.initial_loss_scaling / opts.loss_velocity_scaling_ratio, use_combined_accum=True)
    elif opts.optimizer == 'adamw':
        optimizer = AdamW(params, lr=opts.lr, loss_scaling=opts.initial_loss_scaling, eps=opts.optimizer_eps)
    elif opts.optimizer == 'rmsprop':
        optimizer = RMSprop(params, lr=opts.lr, alpha=opts.rmsprop_decay, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling, eps=opts.optimizer_eps)
    elif opts.optimizer == 'rmsprop_tf':
        optimizer = RMSprop(params, lr=opts.lr, alpha=opts.rmsprop_decay, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling, eps=opts.optimizer_eps, use_tf_variant=True)

    # Make optimizers distributed
    if opts.use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    return optimizer


def get_lr_scheduler(opts, optimizer, step_per_epoch, start_epoch=0):
    scheduler_freq = opts.lr_scheduler_freq if opts.lr_scheduler_freq > 0.0 else step_per_epoch
    scheduler_last_epoch = (scheduler_freq * start_epoch) - 1
    if opts.lr_schedule == "step":
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[step*scheduler_freq for step in opts.lr_epoch_decay], gamma=opts.lr_decay, last_epoch=scheduler_last_epoch)
    elif opts.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=opts.epoch*scheduler_freq, last_epoch=scheduler_last_epoch)
    elif opts.lr_schedule == "exponential":
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=opts.lr_decay, last_epoch=scheduler_last_epoch)

    lr_scheduler = PeriodicLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, period=1./scheduler_freq)
    lr_scheduler = WarmUpLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, warmup_epoch=opts.warmup_epoch)
    return lr_scheduler


if __name__ == '__main__':
    run_opts = parse_arguments()

    logging.info("Loading the data")
    model_opts = create_model_opts(run_opts)
    train_data = datasets.get_data(run_opts, model_opts, train=True, async_dataloader=True)

    logging.info("Initialize the model")
    model = models.get_model(run_opts, datasets.datasets_info[run_opts.data], pretrained=False)
    model.train()

    optimizer = get_optimizer(run_opts, model)

    model_opts = create_model_opts(run_opts)
    model_opts = utils.train_settings(run_opts, model_opts)
    lr_scheduler = get_lr_scheduler(run_opts, optimizer, len(train_data))

    validation_opts = create_validation_opts(run_opts)

    test_data = None
    if run_opts.validation_mode != "none":
        test_data = datasets.get_data(run_opts, validation_opts, train=False, async_dataloader=True, return_remaining=True)

    trainer = pl.Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=20,
        log_every_n_steps=1,
        accumulate_grad_batches=run_opts.gradient_accumulation,
        plugins=IPUPlugin(inference_opts=validation_opts, training_opts=model_opts, autoreport=False)
    )

    model = LightningResnet(model, run_opts.label_smoothing, optimizer, lr_scheduler)

    if test_data is not None:
        trainer.fit(model, train_data)
    else:
        trainer.fit(model, train_data, test_data)

