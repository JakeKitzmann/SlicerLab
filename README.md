# DLMI Slicer Lab

Learning Objectives:

- Explore building a basic U-Net from scratch

- Move existing network weights from training environment to usage

- Build object-oriented 3D Slicer module to run existing U-Net

## Setup

from SlicerLab/:

```bash
module load miniconda

conda create --name "SlicerLab" Python=3.11 -y
conda activate SlicerLab
mkdir -p .pip_cache
export PYTHONNOUSERSITE=1                       # ignore ~/.local packages
export PIP_CACHE_DIR=/nfsscratch/Users/<YOU>/.pip-cache
mkdir -p slicer-cache
mkdir -p slicer-packages
pip install -r requirements.txt
```

Now open the notebook and select SlicerLab

Work through the notebook to train the U-Net and get our weights.

## Slicer

We're not going to use the default slicer, as we want a version we can modify.

Re-Download 3D Slicer with

```bash
curl -fsSL https://download.slicer.org/install.sh | bash
```

Now we can call it locally with:

```bash
Slicer
```

If that doesn't work:

```bash
cd ~/.local/bin 
./Slicer
```

<img width="615" height="314" alt="Screenshot from 2026-11-18 19-30-20" src="https://github.com/user-attachments/assets/e726e743-e23b-4054-9606-a7c410720fa8" />

Open the Slicer Python window (Python Logo in the corner) and enter:

```python
import os
import sys
import slicer

package_dir = "/nfsscratch/Users/<YOU>/SlicerLab/slicer-packages"
cache_dir = "/nfsscratch/Users/<YOU>/SlicerLab/slicer-cache"

os.makedirs(package_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

os.environ["PIP_CACHE_DIR"] = cache_dir

slicer.util.pip_install([
    "--target", package_dir,
    "torch",
    "torchvision",
])
```

In the top left corner of slicer go to:
```
Edit->Application Settings->Modules
```

Now click the two arrows to the right of Additional module paths:

Press Add.

Then add the folder /nfsscratch/Users/<you>/SlicerLab/SlicerLab

Press yes to restart slicer.

## Using the Application

After restarting slicer, your Python console should show the **green text**:

<img width="740" height="399" alt="image" src="https://github.com/user-attachments/assets/57f81bbf-e875-4cdb-bcc6-a7e718b1f4c2" />

If it errors, come grab me as PyTorch hasn't correctly imported.

<img width="740" height="399" alt="image" src="https://github.com/user-attachments/assets/524627a3-bb03-4c6e-854f-4b4905af9095" />

Navigate to "Welcome to Slicer" in the dropdown module selector.

<img width="740" height="399" alt="image" src="https://github.com/user-attachments/assets/a5642ef0-ab6f-4e9e-b1f4-8d9ecf813d07" />
<img width="740" height="399" alt="image" src="https://github.com/user-attachments/assets/eb11e208-18f0-40be-b3c0-e645809c832e" />

Click "Download Sample Data", then click "CTChest".

<img width="214" height="179" alt="image" src="https://github.com/user-attachments/assets/bd6bf4a4-2ca5-409a-8bf6-441f1322c881" />

Go to the "Volumes" module (the box)

<img width="507" height="928" alt="image" src="https://github.com/user-attachments/assets/72d2acdd-d266-4c60-a7a3-7ac53f0f3c96" />

Click on the "CT-Lung" window leveling (just to the left of the RGB ones).

<img width="1561" height="1576" alt="image" src="https://github.com/user-attachments/assets/f3d65c32-8959-4d43-9698-f8e499aa186b" />

Now we have a clearer visualization of the lung tissue.

Back to the modules dropdown, click on DLMI->BME:5240 SlicerLab

Select CT Chest and press "Segment"

<img width="1561" height="1576" alt="image" src="https://github.com/user-attachments/assets/5952fb1b-befc-4dba-9a7c-dfefb9f2c3b6" />

Since we only trained on a few images, the segmentation is rough. Note how the segmentations are more accurate in the Axial plane, as the U-Net only sees one Axial slice at a time.

Go to the Segment Editor module in the top

<img width="251" height="203" alt="image" src="https://github.com/user-attachments/assets/6d39368e-8999-4683-ac98-7cdcb6f355f2" />

Select the segmentation created and click "Show 3D" to create a 3D rendering of the mask created.

<img width="1914" height="1608" alt="image" src="https://github.com/user-attachments/assets/b15f8bb7-d44a-45f0-b687-dedaf2ae261f" />

