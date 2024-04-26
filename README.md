# Embodied Family Code Base

We will update the instructions for this codebase as soon as possible.

## Installation

See [INSTALLATION.md](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch/blob/main/INSTALLATION.md)

## Data Preparation

1. Download the [EgoCOT dataset](https://github.com/EmbodiedGPT/EgoCOT_Dataset).
2. Download the [COCO-2017 dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).

## Download the Pretrained Model

Download the testing
model [Embodied_family_7btiny](https://huggingface.co/Liang-ZX/Embodied_family_7b/).

## Prepare the Text Data Paired with Video and Image

- Unzip `datasets_share.zip`, which contains the text part of the multi-modal dataset, to the `./datasets/` directory.

## 🏠 Overview

<img width="800" alt="image" src="https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch/blob/main/assest/overall_frame_embodiedgpt.png">

## 🎁 Major Features

<img width="800" alt="image" src="https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch/blob/main/assest/main_features_embodiedgpt.png">

## Usage

This repo can be used in conjunction with PyTorch's `Dataset` and `DataLoader` for training models on heterogeneous
data. Here's a brief overview of the classes and their functionalities:

### BaseDataset

The `BaseDataset` class extends PyTorch's `Dataset` and is designed to handle different media types (images, videos, and
text). It includes a transformation process to standardize the input data and a processor to handle the data specific to
the task.

#### Example

```python
from robohusky.base_dataset_uni import BaseDataset

# Initialize the dataset with the required parameters
dataset = BaseDataset(
    dataset,  # Your dataset here
    processor,  # Your processor here
    image_path="path/to/images",
    input_size=224,
    num_segments=8,
    norm_type="openai",
    media_type="image"
)

# Use the dataset with a PyTorch DataLoader
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### WeightedConcatDataset

The `WeightedConcatDataset` class extends PyTorch's `ConcatDataset` and allows for the creation of a unified dataset by
concatenating multiple datasets with specified weights.

#### Example

```python
from robohusky.base_dataset_uni import WeightedConcatDataset

# Assume we have multiple datasets for different tasks
dataset1 = BaseDataset(...)
dataset2 = BaseDataset(...)
dataset3 = BaseDataset(...)

# Define the weights for each dataset
weights = [0.5, 0.3, 0.2]

# Create a weighted concatenated dataset
weighted_dataset = WeightedConcatDataset([dataset1, dataset2, dataset3], weights=weights)

# Use the weighted dataset with a PyTorch DataLoader
data_loader = DataLoader(weighted_dataset, batch_size=32, shuffle=True)
```

## Customization

The package is designed to be flexible and customizable. You can implement your own transformation and processing logic
by subclassing `BaseDataset` and overriding the necessary methods.

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:
```bibtex
@article{mu2024embodiedgpt,
  title={Embodiedgpt: Vision-language pre-training via embodied chain of thought},
  author={Mu, Yao and Zhang, Qinglong and Hu, Mengkang and Wang, Wenhai and Ding, Mingyu and Jin, Jun and Wang, Bin and Dai, Jifeng and Qiao, Yu and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
