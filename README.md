# Subspace Optimization for Backpropagation-Free Continual Test-Time Adaptation

[![Paper](https://img.shields.io/badge/Paper-arXiv:2603.28678-red)](https://arxiv.org/abs/2603.28678)
[![Conference](https://img.shields.io/badge/Conference-ECML%20PKDD%202026-blue)](TODO)

> **Authors:** Damian Sójka, Sebastian Cygert, Marc Masana

## 🛠 Environment Setup

This project targets Python 3.8.18 with `torch==2.1.0` (built for CUDA 11.8). Set it up with `uv`.

1. Install [uv 🔗](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it yet.
2. Create the environment and install dependencies:
   ```bash
   uv sync
   ```
3. Activate it:
   ```bash
   source .venv/bin/activate
   ```
   Alternatively, skip activation and prefix commands with `uv run` (e.g. `uv run python main.py ...`).

## 📊 Dataset Preparation

### ImageNet (clean, `--data`)
The plain ImageNet-1k validation set is required for every run (regardless of which corrupted/shifted dataset is being adapted to): it is used to compute source statistics for ImageNet-based benchmarks.

1. Download the ImageNet-1k ILSVRC2012 validation set from [here 🔗](https://image-net.org/download.php) (registration required).

2. Only the `val` split is used, and it must be arranged into one subfolder per class (the raw ILSVRC2012 val download is a flat directory of images, so it needs to be reorganized first, e.g. with the standard [valprep.sh 🔗](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) script) so it matches `torchvision.datasets.ImageFolder`:

```
imagenet
`-- val
    |-- n01440764
    |-- n01443537
    `-- ...
```

Point `--data` at the `imagenet` folder (i.e. the parent of `val`).

### ImageNet-C (`--data_corruption`)
1. Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

2. Extract the files from the tar archive and organize them in the following format:

```
imagenet-c
|-- brightness
|   |-- 1
|   |-- 2
|   |-- 3
|   |-- 4
|   `-- 5
|-- contrast
|-- defocus_blur
|-- elastic_transform
|-- fog
|-- frost
|-- gaussian_noise
|-- glass_blur
|-- impulse_noise
|-- jpeg_compression
|-- motion_blur
|-- pixelate
|-- shot_noise
|-- snow
`-- zoom_blur
```

Point `--data_corruption` at the `imagenet-c` folder.

### ImageNet-R (`--data_rendition`)
- 1. Download [ImageNet-R 🔗](https://github.com/hendrycks/imagenet-r) dataset from [here 🔗](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). 

- 2. Extract the tar archive. It unpacks directly into one subfolder per class:

```
imagenet-r
|-- n01443537
|-- n01484850
`-- ...
```

No further reorganization is needed — point `--data_rendition` at this extracted `imagenet-r` folder.

### DomainNet-126
- 1. Please download the [DomainNet-126 dataset (cleaned version) 🔗](https://ai.bu.edu/M3SDA/), specifically the `clipart`, `painting`, `real` and `sketch` domains.

- 2. Extract the four domains and organize them under `--dataroot` in the following format:

```
<dataroot>
`-- DomainNet-126
    |-- clipart
    |-- painting
    |-- real
    `-- sketch
```

The .txt files for the image labels are provided under `./dataset/domainnet126_lists/` and already reference this layout (e.g. `real/bird/real_032_000265.jpg`), so no further setup is needed there.

## 🚀 Running Experiments

Experiments are launched through `run_config.py`, which merges [`configs/common.yaml`](configs/common.yaml) with a method-specific config from `configs/<dataset>/<method>.yaml` and runs `main.py` with the resulting arguments.

1. Fill in the dataset paths (`--data`, `--data_corruption`, `--data_rendition`, `--dataroot`) and any general settings (e.g. `arch`, `batch_size`, `quant`) in `configs/common.yaml`.
2. Uncomment and set whichever method-specific hyperparameters you want in `configs/<dataset>/<method>.yaml`, e.g. `configs/imagenet_c/pace.yaml`.
3. Run:
   ```bash
   python run_config.py <method> <dataset>
   ```
   e.g.
   ```bash
   python run_config.py pace imagenet_c
   ```

Available `<method>` values: `tent`, `foa`, `pace`, `t3a`, `sar`, `cotta`, `lame`, `zoa_vit`, `no_adapt`.
Available `<dataset>` values: `imagenet_c`, `imagenet_r`, `domainnet126`.

Any extra `--<arg> <value>` passed after `<method> <dataset>` overrides the corresponding config value for that run, e.g.:
```bash
python run_config.py pace imagenet_c --vector_bank_size 15 --gamma 0.5
```

For experiments with a quantized ViT, set `quant: true` in `configs/common.yaml` (or pass `--quant` as an override).

## 🤖 Source Model Checkpoints
Checkpoints for DomainNet-126 are available [here 🔗](https://drive.google.com/drive/folders/1DdhSbbxoHBtF_5gyvZ5gBN6KYxknK6-s?usp=sharing). Download the ckpts folder and place it at the repository root.

Checkpoint for ImageNet-C/R datasets are automatically downloaded.

## 📄 Citation
```bibtex
@inproceedings{sojka2026subspace,
  title         = {Subspace Optimization for Backpropagation-Free Continual Test-Time Adaptation},
  author        = {S{\'o}jka, Damian and Cygert, Sebastian and Masana, Marc},
  booktitle     = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  year          = {2026},
}
```

## Acknowledgment

The code is inspired by [FOA 🔗](https://github.com/mr-eggplant/FOA.git) and [ZOA 🔗](https://github.com/DengZeshuai/ZOA.git).