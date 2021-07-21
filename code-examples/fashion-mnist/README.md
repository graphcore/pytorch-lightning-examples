
# ResNet18

This script showcases a FashionMNIST example using either a handwritten ResNet18 model or an off the shelf Torchvision backbone running on the Graphcore IPU, with PyTorch Lightning.

ResNet18 was designed for the ImageNet dataset, which classifies images into 1000 categories. This example uses the FashionMNIST dataset which classifies images into only 10 categories so a few edits have been made to the model to address this.

To run the script execute:

```console
python3 train.py
```

To toggle between the two models provide the option --model MODEL (either "ours" or "torchvision"), default "ours".

```console
python3 train.py --model torchvision
```