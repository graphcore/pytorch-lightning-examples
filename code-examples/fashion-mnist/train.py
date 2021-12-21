# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import torchvision
import poptorch
import pytorch_lightning as pl
from pytorch_lightning.plugins import IPUPlugin
from torch import nn
import torchvision.models as models
import argparse

# Basic block with residual connections.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_inp=(1,1), downsample_key=False):
        super().__init__()
        self.block = torch.nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride_inp, padding=(1,1), bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=(1,1), bias=False),
                    nn.BatchNorm2d(out_channels))

        self.downsample = None
        if downsample_key:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride_inp, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.block(x)
        if self.downsample: out += self.downsample(x)
        return out


# A handwritten ResNet which also outputs 10 classes.
class ResNetFromScratch(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(nn.Conv2d(1, 64, 7), nn.BatchNorm2d(64),nn.ReLU(),
                                         nn.MaxPool2d(3,2,1),
                                            nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64)),
                                            nn.Sequential(ResidualBlock(64, 128, (2,2), True), ResidualBlock(128, 128)),
                                            nn.Sequential(ResidualBlock(128, 256, (2,2), True), ResidualBlock(256, 256)),
                                            nn.Sequential(ResidualBlock(256, 512, (2,2), True), ResidualBlock(512, 512)),
                                 nn.AdaptiveAvgPool2d((1,1)),torch.nn.Flatten(), nn.Linear(512, 10), nn.LogSoftmax(1))

    def forward(self, x):
        x = self.model(x)
        return x

# Takes an off the shelf torchvision model and modifies it for 10 FashionMNIST classes.
class TorchVisionBackbone(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18()

        # Overwriting the imported model's Conv1 layer to change it from 3 RGB channels to FashionMNIST's 1.
        self.network.conv1 = nn.Conv2d(1, 64, 7)

        # Overwriting the imported model's FC layer
        num_features = self.network.fc.in_features
        self.network.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.network(x)
        x = torch.nn.functional.log_softmax(x)
        return x

# ResNet18 from-scratch with some changes to suit the FashionMNIST dataset
class ResNetClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        output = self.forward(x)
        loss = torch.nn.functional.nll_loss(output, y)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        output = self.forward(x)
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds==y).float() / len(y)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



class FashionMNIST(pl.LightningDataModule):
    def __init__(self, options, batch_size=4):
        super().__init__()
        self.batchsize = batch_size
        self.options = options

    def setup(self, stage="train"):
        # Retrieving the datasets
        self.train_data = torchvision.datasets.FashionMNIST(
            'fashionmnist_data/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307, ),
                    (0.3081, )
                )
            ]))

        self.validation_data = torchvision.datasets.FashionMNIST(
            'fashionmnist_data/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307, ),
                    (0.3081, )
                )
            ]))

    def train_dataloader(self):
        return poptorch.DataLoader(
            dataset=self.train_data,
            batch_size=self.batchsize,
            options=self.options,
            shuffle=True,
            drop_last=True,
            mode=poptorch.DataLoaderMode.Async,
            num_workers=32
        )

    def val_dataloader(self):
        return poptorch.DataLoader(
            dataset=self.validation_data,
            batch_size=self.batchsize,
            options=self.options,
            drop_last=True,
            mode=poptorch.DataLoaderMode.Async,
            num_workers=32
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='The ResNet model to use. Either \"ours\" or \"torchvision\"',
                        type=str,
                        default="ours")
    args = parser.parse_args()


    if args.model == "ours":
        model =ResNetFromScratch()
    elif args.model == "torchvision":
        model = TorchVisionBackbone()
    else:
        print("Invalid model selection: \"%s\", options are \"ours\" or \"torchvision\"" % args.model)
        exit()


    # Initializing the model
    model = ResNetClassifier(model)

    options = poptorch.Options()
    options.deviceIterations(250)
    options.replicationFactor(8)

    datamodule = FashionMNIST(options)

    trainer = pl.Trainer(
        max_epochs=20,
        progress_bar_refresh_rate=20,
        log_every_n_steps=1,
        plugins=IPUPlugin(inference_opts=options, training_opts=options)
    )

    trainer.fit(model, datamodule)