# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import torch
import poptorch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.plugins import IPUPlugin
from simple_lightning_ipu import SimpleLightning




if __name__ == '__main__':
    # Create the model as usual.
    model = SimpleLightning()

    # Normal PyTorch dataset.
    data_set = torchvision.datasets.FashionMNIST("FashionMNIST",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor()]))


    # PopTorch includes an Options class which exposes additional IPU specific hardware and software options.
    # See the documentation on [session control options]
    # (https://docs.graphcore.ai/projects/popart-user-guide/en/latest/importing.html#session-control-options)
    # and [batching]
    # (https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#efficient-data-batching)
    # for further information on specific options.


    # Firstly we start by creating the options class. You can have one set for training and one for inference.
    train_options = poptorch.Options()
    validation_options = poptorch.Options()


    # One useful option is device iterations. This allows the IPU to eliminate some
    # host overhead by pulling more elements from the dataloader at a time while
    # still running with the normal model batchsize.
    train_options.deviceIterations(300)
    validation_options.deviceIterations(250)


    # Replication factor will replicate the program across multiple IPUs
    # automatically. This can also be done using the ipus=N option. However the
    # dataloader will not automatically pull in the additional elements leading
    # to it pulling in batch_size/replication factor elements at a time.
    train_options.replicationFactor(8)
    validation_options.replicationFactor(4)


    # To avoid this we provide a poptorch.Dataloader class. This is almost the
    # same as `torch.utils.data.DataLoader` which it wraps. The difference
    # being it takes in a `poptorch.Options` class which is then used to calculate
    # the correct batchsize to pull in.
    valid_data = poptorch.DataLoader(validation_options,
        data_set, batch_size=16, shuffle=True)


    # It also supports a `poptorch.DataLoaderMode.Async` which will load in the data
    # asynchronously to further reduce host overhead.
    train_data = poptorch.DataLoader(train_options,
        data_set, batch_size=16, shuffle=True, mode=poptorch.DataLoaderMode.Async)


    # PyTorch Lightning provides an `IPUPlugin` class which takes in these options
    # and will pass them to PopTorch under the hood.
    trainer = pl.Trainer(max_epochs=1,
                        progress_bar_refresh_rate=20,
                        log_every_n_steps=1,
                        plugins=IPUPlugin(inference_opts=validation_options, training_opts=train_options)
                        )

    trainer.fit(model, train_data, valid_data)
