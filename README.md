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
mkdir -p pip-cache
python -m pip config set global.cache-dir ./pip-cache
pip install -r requirements.txt
python -m ipykernel install --user --name SlicerLab --display-name "SlicerLab"
```

Now open the notebook and select SlicerLab
