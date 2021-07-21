# Small tutorial examples

This directory contains some small code examples showing a series of concepts in a minimal amount of code.


Each example can be run as follows:

```console
python3 NAME.py
```

The comments in each one describe what it is doing/showing.

There is a simple model which we use to create the examples.
* `simple_torch_model.py` A simple model for us to use in the examples.

The examples show:
* `simple_lightning_ipu.py` Shows how to run a simple Lightning model on IPU.
* `ipu_options_and_dataloading.py` Shows how to pass PopTorch options. Includes tutorial on replication and dataloading.
* `pipelined_ipu.py` Shows how to pipeline a model across IPUs.
* `custom_losses.py` Shows how to add a custom loss on the IPU in a simple regression model.
