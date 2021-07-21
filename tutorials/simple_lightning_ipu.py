# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import torch

import torchvision
import torchvision.transforms as transforms

from simple_torch_model import SimpleTorchModel


# This class shows a minimal lightning example. This example uses our own
# SimpleTorchModel which is a basic 2 conv, 2 FC torch network. It can be
# found in simple_torch_model.py.
class SimpleLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleTorchModel()

    def training_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        loss = torch.nn.functional.nll_loss(prediction, label)
        return loss


    def validation_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        preds = torch.argmax(prediction, dim=1)
        acc = torch.sum(preds==label).float() / len(label)
        return acc

    # PopTorch doesn't currently support logging within steps. Use the Lightning
    # callback hooks instead.
    def on_train_batch_end(self,outputs, batch, batch_idx, dataloader_idx):
        self.log('StepLoss', outputs["loss"])

    def validation_epoch_end(self, outputs):
        self.log('val_acc', torch.stack(outputs).mean(), prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == '__main__':
    # Create the model as usual.
    model = SimpleLightning()

    # Normal PyTorch dataset.
    train_set = torchvision.datasets.FashionMNIST("FashionMNIST",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor()]))

    # Normal PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=16,
                                            shuffle=True)

    # Run on IPU using IPUs=1. This will run on IPU but will not include any custom
    # PopTorch Options. Changing IPUs to 1 to IPUs=N will replicate the graph N
    # times. This can lead to issues with the DataLoader batching - the script
    # ipu_options_and_dataloading.py shows how these can be avoided through the
    # use of IPUOptions.
    trainer = pl.Trainer(ipus=1,
                        max_epochs=3,
                        progress_bar_refresh_rate=20,
                        log_every_n_steps=1)


    # When fit is called the model will be compiled for IPU and will run on the available IPU devices.
    trainer.fit(model, train_loader)
