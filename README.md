# Graphcore code examples and tutorials for PyTorch Lightning

This repository contains a number of sample applications and tutorials specifically for the PyTorch Lightning IPU integration.

The content of this repository requires Poplar SDK version 2.1 and PyTorch Lightning version 1.4 or higher.

Before running these examples, you need to install the Poplar SDK following the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system. Make sure to source the enable.sh scripts for Poplar and PopART. We recommend that you install PopTorch in a Python virtualenv.

We have three categories of examples.

* [tutorials](https://github.com/graphcore/pytorch-lightning-examples/tree/release/tutorials) Short snippets each showing how to use one particular feature.
* [code-examples](https://github.com/graphcore/pytorch-lightning-examples/tree/release/code-examples/) Contains two examples showing Lightning/PopTorch integration.
    * [shakespeare-rnn](https://github.com/graphcore/pytorch-lightning-examples/tree/release/code-examples/shakespeare-rnn) A port of an existing PyTorch RNN to run on IPU through Lightning.
    * [fashion-mnist](https://github.com/graphcore/pytorch-lightning-examples/tree/release/code-examples/fashion-mnist) Runs two different ResNets, one a modified torchvision backbone the other written from scratch
* [applications](https://github.com/graphcore/pytorch-lightning-examples/tree/release/applications) A port of the [Convolutional Neural Network training application](https://github.com/graphcore/examples/tree/master/applications/pytorch/cnns/train) from Graphcore's [examples](https://github.com/graphcore/examples) repo using Lightning. This can be used to run, and benchmark, many CNN examples examples.
