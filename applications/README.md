# PyTorch CNN application

This directory contains a PyTorch Lightning port of the [PyTorch CNNs training application from the Graphcore GitHub examples repository](https://github.com/graphcore/examples/tree/master/applications/pytorch/cnns/train).

It can be used to verify that the same performance is obtained with PyTorch Lightning as when running directly in PopTorch.

To run the CNN application using PyTorch Lightning:

```
git submodule init
git submodule update
pip3 install ./examples/applications/pytorch/cnns/requirements.txt
```

See the README.md in examples/applications/pytorch/cnns for full documentation, the
following run command and options are repeated here for ease of use.

```
       python3 train_lightning.py --data imagenet --imagenet-data-path <path-to/imagenet>
```

#### ImageNet

|IPU configuration|Model  | Config name| Note |
|-------|----------|---------|---------|
|Mk2 IPU-POD16|ResNet50| `resnet50_mk2_pipelined`| 4 pipeline stages, 4 replicas |
|Mk2 IPU-POD16|ResNet50| `resnet50-16ipu-mk2-recompute`| single ipu, 16 replicas |
|Mk2 IPU-POD16|EfficientNet-B0| `efficientnet-b0-16ipu-mk2`|4 pipeline stages, 4 replicas |


```
python3 train_lightning.py --config <config name> --imagenet-data-path <path-to/imagenet>
```

## Options

The program has a few command line options:

`-h`                            Show usage information

`--config`                      Apply the selected configuration

`--seed`                        Provide a seed for random number generation

`--batch-size`                  Sets the batch size for training

`--model`                       Select the model (from a list of supported models) for training

`--data`                        Choose the dataset between `cifar10`, `imagenet`, `generated` and `synthetic`. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is generated on the device. In generated mode random data is created on host side.

`--imagenet-data-path`          The path of the downloaded ImageNet dataset (only required if imagenet is selected as data)

`--pipeline-splits`             List of layers to create stages of the pipeline. Each stage runs on different IPUs. Example: layer0 layer1/conv layer2/block3/bn

`--replicas`                    Number of IPU replicas

`--device-iterations`           Sets the device iteration: the number of inference steps before program control is returned to the host

`--precision`                   Precision of Ops(weights/activations/gradients) and Master data types: `16.16`, `32.32` or `16.32`

`--half-partial`                Flag for accumulating matrix multiplication partials in half precision

`--available-memory-proportion` Proportion of memory which is available for convolutions

`--gradient-accumulation`       Number of batches to accumulate before a gradient update

`--lr`                          Initial learning rate

`--epoch`                       Number of training epochs

`--norm-type`                   Select the used normlayer from the following list: `batch`, `group`, `none`

`--norm-num-groups`             If group normalization is used, the number of groups can be set here

`--full-precision-norm`         Calculate the norm layers in full precision.

`--enable-fast-groupnorm`       There are two implementations of the group norm layer. If the fast implementation enabled, it couldn't load checkpoints, which didn't train with this flag. The default implementation can use any checkpoint.

`--disable-stable-batchnorm`    There are two implementations of the batch norm layer. The default version is numerically more stable. The less stable is faster.

`--validation-mode`             The model validation mode. Possible values are `none` (no validation) `during` (validate after every n epochs) and `after` (validate after the training).

`--validation-frequency`        How many training epochs to run between validation steps.

`--disable-metrics`             Do not calculate metrics during training, useful to measure peak throughput

`--enable-recompute`            Enable the recomputation of network activations during backward pass instead of caching them during forward pass. This option turns on the recomputation for single-stage models. If the model is multi-stage (pipelined) the recomputation is always enabled.

`--recompute-checkpoints`       List of recomputation checkpoints. List of regex rules for the layer names. (Example: Select convolutional layers: `.*conv.*`)

`--offload-optimizer`           Store the optimizer status off-chip

`--lr-schedule`                 Select learning rate schedule from [`step`, `cosine`, `exponential`] options

`--lr-decay`                    Learning rate decay (required with step schedule). At the predefined epoch, the learning rate is multiplied with this number

`--lr-epoch-decay`              List of epochs, when learning rate is modified

`--warmup-epoch`                Number of learning rate warmup epochs

`--checkpoint-path`             The checkpoint folder. In the given folder a checkpoint is created after every epoch

`--optimizer`                   Define the optimizer: `sgd`, `sgd_combined`, `adamw`, `rmsprop`, `rmsprop_tf`

`--momentum`                    Momentum factor

`--optimizer-eps`               Small constant added to the updater term denominator for numerical stability.

`--loss-scaling`                Loss scaling factor. This value is reached by the end of the training.

`--initial-loss-scaling`        Initial loss scaling factor. The loss scaling interpolates between this and loss-scaling value. The loss scaling value multiplies by 2 during the training until the given loss scaling value is not reached. If not determined the `--loss-scaling` is used during the training. Example: 100 epoch, initial loss scaling 16, loss scaling 128: Epoch 1-25 ls=16;Epoch 26-50 ls=32;Epoch 51-75 ls=64;Epoch 76-100 ls=128

`--enable-stochastic-rounding`  Enable Stochastic Rounding

`--weight-decay`                Weight decay factor

`--wandb`                       Use Weights and Biases to log the training

`--logs-per-epoch`              Number of logging steps in each epoch

`--label-smoothing`             The label smoothing factor: 0.0 means no label smoothing (this is the default)

`--lr-scheduler-freq`           Number of learning rate update in each epoch. In case of 0 used, it is updated after every batch

`--weight-avg-strategy`         Weight average strategy

`--weight-avg-exp-decay`        The exponential decay constant for weight averaging. Applied if exponential weight average strategy is chosen

`--weight-avg-N`                Weight average applied on last N checkpoint, -1 means all checkpoints

`--rmsprop-decay`               RMSprop smoothing constant

`--efficientnet-expand-ratio`   Expand ratio of the blocks in EfficientNet

`--efficientnet-group-dim`      Group dimensionality of depthwise convolution in EfficientNet

`--profile`                     Generate PopVision Graph Analyser report

`--loss-velocity-scaling-ratio` Only for sgd_combined optimizer: Loss Velocity / Velocity scaling ratio. In case of large number of replicas >1.0 can increase numerical stability

`--use-bbox-info`               Images may contain bounding box information for the target object. If this flag is set, during the augmentation process make sure the augmented image overlaps with the target object.

`--eight-bit-io`                Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU

`--normalization-location`      Location of the data normalization. Options: `host`, `ipu`, `none`

`--dataloader-rebatch-size`     Batch size of the dataloader worker. The final batch is created from the smaller batches. Lower value results in less host-side memory. A higher value can reduce the overhead of rebatching. This setting can be useful for reducing memory pressure with a large global batch size.

`--iterations`                  Number of program iterations for generated and synthetic data. This helps to modify the length of these datasets.

`--model-cache-path`            If path is given the compiled model is cached to the provided folder.
