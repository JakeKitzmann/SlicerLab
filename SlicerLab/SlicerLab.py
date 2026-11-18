import sys

PACKAGE_DIR = "/nfsscratch/Users/jkitzmann/SlicerLab/slicer-packages"

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

import os
import slicer
import qt
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

import SimpleITK as sitk
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", device)

class SlicerLab(ScriptedLoadableModule):

    def __init__(self, parent):
        super().__init__(parent)

        self.parent.title = "BME:5240 Slicer Lab"
        self.parent.categories = ["DLMI"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "<YOU>",
            "Jake Kitzmann",
            "Sarah Gerard",
        ]
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
        self.runButton.connect("clicked(bool)", self.onRun)

        self.layout.addStretch(1)

    def onRun(self):
        volume = self.volumeSelector.currentNode()

        if volume is None:
            slicer.util.errorDisplay("Select a volume to segment.")
            return

        print("Segmenting...")


