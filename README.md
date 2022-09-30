# Bootstrapping the Relationship Between Images and Their Clean and Noisy Labels

The code for our WACV 2023 submission "Bootstrapping the Relationship Between Images and Their Clean and Noisy Labels".

## Installation

`environment.yml` contains the Anaconda environment used to run the experiments, with some of the packages having a Linux dependency.

The pretrained model weights, and saved noisy labels can be generated using the available code, or downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1IFBh5kVIFVl4HetrRDMoyvWXdARlgVyU?usp=sharing). Once downloaded, the `datasets.zip` and `models.zip` can be extracted and used to replace the `/datasets` and `/models` folders respectively.

## Preparing Data

Data is stored in the '/data' folder in the root directory of this project.
- CIFAR10 and CIFAR100 should download automatically into the `/data/cifar10` and `/data/cifar100` folders.
- Animal10N can be downloaded following the instructions [here](https://dm.kaist.ac.kr/datasets/animal-10n/), and should be placed into `/data/animal10n` such that `/data/animal10n/training` contains the training images and `/data/animal10n/testing` contains the testing images
- Red Mini-Imagenet can be downloaded following the instructions [here](https://google.github.io/controlled-noisy-web-labels/index.html). We perform our tests using only Red Mini-Imagenet. We prepared the dataset by placing the split files in `data/red_blue/split`, downloading the images from their URIs to `data/red_blue/all_images`, and downloading the validation images belong to `<class_no>` to `data/red_blue/<class_no>`. The dataset in `datasets/red_blue/red_blue.py` can be easily modified to support other configurations of the data.
- Webvision can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html), and images from the first 50 classes should be resized to 256x256 and placed in `/data/webvision/val_images_256`. Additionally, `train_filelist_google.txt` should be downloaded from [the Google Drive link](https://drive.google.com/drive/folders/1IFBh5kVIFVl4HetrRDMoyvWXdARlgVyU?usp=sharing) and placed in `/datasets/webvision/info/train_filelist_google.txt`, and `val_filelist.txt` should be similarly downloaded and put in `/datasets/webvision/info/val_filelist.txt`. For ILSVRC2012 validation, the relevant validation samples should be resized to 256x256 and placed in `/data/ILSVRC2012_val/`, and `ILSVRC2012_validation_to_webvision_labels_ground_truth.txt` from the Google Drive link should be placed in `datasets/ilsvrc2012/ILSVRC2012_validation_to_webvision_labels_ground_truth.txt`.

## Preparing Noisy Labels

If `datasets.zip` was downloaded from the Google drive and extracted to `/datasets`, no further action is needed. Otherwise, the dataset files need to be setup by running `python run_ablations setup_<dataset>_datasets 0` where `<dataset>` should be replaced with `cifar10`, `cifar100`, `animal10n`, `red_blue` or `webvision` to setup the respective datasets.

## Preparing Pretrained Models

If `models.zip` was downloaded from the Google drive and extracted to `/models`, no further action is needed. Otherwise, the pretrained models need to be trained using SimCLR or MocoV2 as described in the paper. We don't provide specific exact code for reproducing these pretrained models, however the SimCLR results on CIFAR datasets were trained using the code in `/utils/simclr`, and the hyperparameters for all self-supervised training are outlined in the paper.

Once trained, the SimCLR models should be placed in `models/simclr_pretrained/<dataset>_<model>.pth` (eg. `models/simclr_pretrained/cifar10_preact_resnet18.pth`) and the Webvision Inception ResnetV2 model trained with MocoV2 should be placed in `models/mocov2_pretrained/webvision.pth`.

## Reproducing Results

The main experiments in our paper can be reproduced by running `python run_ablations.py <experiment_name> <gpu_id>`. See `run_ablations.py` for a list of experiments that can be run.

We have attempted to seed our experiments, however we note that differences in hardware may lead to different results. We ran our experiments on machines using RTX 2080Tis and RTX 3090s.

## Code Adaptions

Sections of the code have been adapted from [Divide-And-Co Training](https://github.com/mzhaoshuai/Divide-and-Co-training) and [TorchSSL](https://github.com/TorchSSL/TorchSSL)