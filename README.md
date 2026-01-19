# Exploring the Generation of Synthetic Multiple Sclerosis Lesions in Healthy 3D Brain MRIs Using GANs 

This repository contains the code used in my Master’s thesis to generate **semi-synthetic 3D brain MRIs with realistic multiple sclerosis (MS) white matter lesions**, starting from **healthy, lesion-free scans**.

The approach connects **two Pix2Pix GANs** in an inference pipeline:

1. **GAN1** generates a **synthetic lesion mask** (lesion placement/shape) from a **white matter mask** derived from healthy MRIs.  
2. **GAN2** inserts **realistic-appearing lesions** into a healthy **FLAIR** scan conditioned on the GAN1 mask.

![Pipeline overview](pipeline_overview.png)

This project is based on **Pix2PixNIfTI** (Pix2Pix extended to 2D/3D NIfTI volumes):
- Upstream: https://github.com/giuliabaldini/Pix2PixNIfTI  
- Original Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  

> **Note:** This is research/thesis code. The scripts are provided as used in the thesis and may require adaptation
> (paths, folder layout, cluster settings) to run on a different system or dataset.

---

## Key Features

- **Two-stage lesion synthesis pipeline (GAN1 → GAN2)**
  - GAN1: *white matter mask* → *synthetic MS lesion mask*
  - GAN2: *(healthy FLAIR + synthetic lesion mask)* → *FLAIR with synthetic lesions*

- **3D NIfTI support** via the Pix2PixNIfTI codebase (training/inference on volumetric MRI data)

- **Direct output pairing (image + mask)**
  Produces a semi-synthetic MRI *and* its corresponding lesion mask (from GAN1), enabling downstream tasks without additional manual labeling.

- **Task-specific dataset handling**
  Additional dataset classes for the two GAN stages (see `data/gan1_dataset.py` and `data/gan2_dataset.py`).

- **Experiments & pipeline tooling**
  The `experiments/` folder contains the full set of scripts used in this project, including:
  - **preprocessing** utilities (shared and GAN-specific),
  - **training and hyperparameter search** for GAN1 and GAN2,
  - **inference pipeline** scripts (GAN1 → overlay → GAN2),
  - **evaluation/metrics** computation,
  - and **Slurm job scripts** to run all stages on an HPC cluster.


## Changes made to Pix2PixNIfTI

Compared to the upstream Pix2PixNIfTI repository, the main project-specific changes are:

- Added **GAN1/GAN2 dataset modes** (`data/gan1_dataset.py`, `data/gan2_dataset.py`)
- Modified training/testing entry points:
  - updated `train.py` and `test.py`
  - added `train_gan2.py` (largely based on `train.py`, with GAN2-specific adjustments)
- Modified **loss functions** for GAN1 and GAN2:
  - `models/pix2pix3d_model.py`
  - `models/pix2pix_model.py`
- Added the complete thesis workflow under **`experiments/`**

## Installation & Quick Start

This repository can be used in **two modes**:

1) **Local (Python venv):** preprocessing + evaluation 
2) **HPC Computing Cluster (Slurm + Apptainer):** training + testing

Both workflows use the same codebase (most entry points live under `experiments/`).
Start with preprocessing scripts, then use slurm scripts for training and model testing.

---

## 1) Local workflow (Preprocessing & Evaluation): Python venv

Preprocessing and results evaluation were run locally in a virtual environment.

### Requirements (local)

- Python `3.12.3`
- Dependencies from `requirements.txt`

### Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 
### Preprocessing (required before model training)

Preprocessing is implemented as **Python scripts** under `experiments/` (not fully automated via CLI).
You will need to **edit file paths and, if necessary, filename patterns** in the scripts to match your local dataset
layout.

- Shared preprocessing logic: `experiments/base_preprocessing.py` (`ImgPreprocessing`)
- GAN1 preprocessing entry point: `experiments/gan1_hyperparam_training/gan1_preprocess_training_data.py`
- GAN2 preprocessing entry point: `experiments/gan2_hyperparam_training/preprocess_gan2_training_data.py`

#### FreeSurfer prerequisite 

GAN1 preprocessing relies on a **FreeSurfer `recon-all` run per subject** (using the **T1** image) to obtain
`aseg.mgz`, which is used to derive a **white matter mask**.

> `recon-all` is **time-intensive**. Plan this step accordingly.

#### Run preprocessing

```bash
cd experiments/gan1_hyperparam_training
python gan1_preprocess_training_data.py
```
---

## 2) Cluster workflow (Training & Testing)

Training and testing of GANs in this project was developed and executed primarily on an HPC computing cluster
using Slurm.
The original upstream project ships a `Dockerfile`. In this thesis project, experiments were run on an HPC cluster using
**Apptainer** (formerly **Singularity**) 

- If you have **Docker** available (local workstation/server): you can build and run the provided `base.dockerfile`.
- If you are on an **HPC system with Apptainer**: you can use **Apptainer** to build/run an image *from the Dockerfile*.

#### Requirements (recommended setup)

- **Linux** environment
- **NVIDIA GPU** with CUDA support (for training/inference)
  - *Note:* GPU memory requirements depend on model/configuration (2D vs 3D, patch size, batch size).  
    Start with the provided scripts and reduce batch size / volume size if you hit OOM.
- **Apptainer** / **Docker**
- **Slurm** (required for using the provided sbatch scripts)


#### Container environment (recommended)

The container is based on `nvcr.io/nvidia/pytorch:21.02-py3` and installs the required Python packages via Conda/pip
(e.g. `nibabel`, `torchio`, `opencv-python`, `visdom`).

##### Build (Docker)
```bash
docker build -f base.dockerfile -t ms-lesion-pix2pix .
```

##### Build (Apptainer/Singularity)
```bash
apptainer build ms-lesion-pix2pix.sif base.dockerfile
``` 

Most workflows can be launched by submitting the
provided `sbatch` scripts under `experiments/**/slurm_scripts/`.

#### Example execution (GAN1 training):

Before submitting any job in `experiments/**/slurm_scripts/`, you must adapt the hard-coded paths at the top of the
scripts. These typically include:

- container image path (`IMAGE=...`) and working directory (`WORKDIR=...`)
- input/output directories for sources, registered images, results, and CSVs

Then submit the job via:

```bash
sbatch experiments/gan1_hyperparam_training/slurm_scripts/train_gan1_diff_hyperparams.sh
``` 

---

## Data

- The datasets/images used for the experiments are listed in [`datasets_report.md`](datasets_report.md).
- The required folder structure is **not fully generalized**. It is implicitly defined in the preprocessing scripts
  under `experiments/` (in particular `gan1_preprocess_training_data.py` and `preprocess_gan2_training_data.py`).
  You will likely need to adapt paths and naming conventions to match your local data layout.
- The datasets are not included in this repository. All datasets used are **publicly available**, however some of them require an **access request / data usage agreement** before download.

---

## License

See [`LICENSE`](LICENSE).

This repository is based on and includes code derived from the upstream projects **Pix2PixNIfTI** and
**pytorch-CycleGAN-and-pix2pix**. Their original licenses and copyright notices apply to the corresponding parts of the code.
Please also refer to:
- https://github.com/giuliabaldini/Pix2PixNIfTI
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix