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

