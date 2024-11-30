# Dealing with Synthetic Data Contamination in Online Continual Learning

Official implementation of our paper titled "[Dealing with Synthetic Data Contamination in Online Continual Learning](https://neurips.cc/virtual/2024/poster/95581)". The paper has been accepted by NeurIPS 2024.

[![arXiv](https://img.shields.io/badge/arXiv-2411.13852-b31b1b.svg)](https://arxiv.org/abs/2411.13852)

## 0. Notes
This codebase is still under further polish. There are still temp hanging variables in the implementation. We name ESRM as "Ours" in the implementation, and we will update the further polished code soon.

## 1. Datasets
### CIFAR-10/100
Torchvision should be able to handle the CIFAR-10/100 dataset automatically. If not, please download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in the `data` folder.

### TinyImageNet
This codebase should be able to handle TinyImageNet dataset automatically and save them in the `data` folder. If not, please refer to [this github gist](https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54).

### ImageNet-100
Download the ImageNet dataset from [here](http://www.image-net.org/) and follow [this](https://github.com/danielchyeh/ImageNet-100-Pytorch) for ImageNet-100 dataset generation. Put the dataset in the `imagenet100_data` folder. Symbolic links are highly recommended.

### Synthetic-C10/C100/Tiny/In-100

TODO: We will share the synthetic images we generated soon.

For Stable Diffusion v1.4/v2.1, Stable Diffusion XL and VQDM, we use pipelines from the [Diffuser](https://huggingface.co/docs/diffusers/en/index) liberary. For GLIDE, we follow [this notebook](https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb). Examples using synthetic models to generate SDXL-C10/GLIDE-C10 can be found in `./syn_data`. Please note to adjust the output directory in `get_path` function. The directory containing synthetic images is expected to be like:

```bash
/home/User/<PATH/TO/SYNTHETIC/DATA>
├── glide   # generation method
├── vqdm
├── v1.4
├── v2.1
└── xl
    ├── 0       # class index          
    │   ├── 1.png       # synthetic images
    │   ├── 2.png
    │   ├── ...
    │   └── n.png
    ├── 1       
    ├── ...       
    └── n       
```

For ImageNet-100 synthetic images, we use pytorch bulid-in [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) and the folder name is expected to match official ImageNet implementation (i.e. like "n01558993").

## 2. Reproduce our results
We use the following hardware and software for our experiments:
- Hardware: NVIDIA Tesla A100 GPUs
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## 3. Training
Weight and bias is highly recommended to run the training, and some features in this codebase is only available with weight and bias. However, it is possible to excute the training without weight and bias.

### Training with a configuration file
Training can be done by specifying the dataset path and params in a configuration file, for example:

```
python main.py --data-root ./data --config ./config/neurips24/ER,c100.yaml
```

Since We do not use this method for training, we only give some configuration examples in `./config/neurips24`. We highly suggest to use weight and bias for the training.

### Training with weight and bias sweep (Recommended)
Weight and bias sweep is originally designed for hyperparameter search. However, it make the multiple runs much easier. Training can be done with W&B sweep more elegantly, for example:

```
wandb sweep sweeps/neurips24/ER,cifar10.yaml
```

Note that you need to set the dataset path in .yaml file by specify `syn_root` (for synthetic dataset) and `--data-root-dir`(for real dataset). And run the sweep agent with:

```
wandb agent $sweepID
```

The hyperparameters after our hyperparameter search is located at `./sweeps/neurips24`.

## 4. Model / memory buffer snapshots
We save the model and memory buffer status after training for evaluation. After the training process, the model should be saved at `./checkpoints/$dataset/$learner/$memory_size/$rand_seed/model.pth` and memory buffer should be located at  `./checkpoints/$dataset/$learner/$memory_size/$rand_seed/memory.pkl`.  

## Acknowledgement
Special thanks to co-author Nicolas. Our implementation is based on his work [AGD-FD's codebase](https://github.com/Nicolas1203/ocl-fd).