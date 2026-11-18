import importlib.util
import slicer
import qt
import vtk

# imports 
pkgs = {
    "SimpleITK": "SimpleITK",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
}

missing = [package for module, package in pkgs.items() if importlib.util.find_spec(module) is None]

if missing:
    print('installing missing pkg:', missing)
    slicer.util.pip_install(missing)

    slicer.util.errorDisplay('Dependencies installed. Please restart Slicer') 
    raise RuntimeError('Restart Slicer to load dependencies')

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget
)

import SimpleITK as sitk
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# change to mps if on mac
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SlicerLab(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "BME:5240 Slicer Lab"
        self.parent.categories = ["DLMI"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name", "Jake Kitzmann", "Sarah Gerard"]
        self.parent.helpText = "DLMI Lab"

class SlicerLabWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)

        self.layout.addWidget(qt.QLabel("Input Volume"))
        self.layout.addWidget(self.volumeSelector)
        
        self.runButton = qt.QPushButton("Segment")
        self.layout.addWidget(self.runButton)
        self.runButton.connect('clicked(bool)', self.onRun)

        self.layout.addStretch(1)

    def onRun(self):
        volume = self.volumeSelector.currentNode()

        if volume is None:
            slicer.util.errorDisplay("Select a volume to segment.")
            return

        print('Segmenting...')
        
class UNet
