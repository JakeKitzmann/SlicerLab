# DLMI Slicer Lab

Learning Objectives:

- Explore building a basic U-Net from scratch

- Move existing network weights from training environment to usage

- Build object-oriented 3D Slicer module to run existing U-Net

## Setup

from SlicerLab/:

```bash
module load miniconda

conda create --name "SlicerLab"
conda activate SlicerLab
mkdir -p .pip_cache
export PIP_CACHE_DIR="$(pwd)/.pip_cache"
pip install -r requirements.txt
python -m ipykernel install --user --name SlicerLab --display-name "SlicerLab"
```

Now open the notebook and select SlicerLab

Work through the notebook to train the U-Net and get our weights.

## Slicer

We're not going to use the default slicer, as we want a version we can modify.

Re-Download 3D Slicer with

```bash
curl -fsSL https://download.slicer.org/install.sh
```


