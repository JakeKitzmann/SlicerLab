import sys

import os
PACKAGE_DIR = "/nfsscratch/Users/jkitzmann/SlicerLab/slicer-packages"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_unet2d.pth")

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

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

        # Load model
        self.model = UNet2D()  
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded on {device}")

        # Existing UI setup
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

        if "model" not in globals():
            slicer.util.errorDisplay("Model has not been loaded.")
            return

        print("Segmenting...")

        image_array = slicer.util.arrayFromVolume(volume)
        image = sitk.GetImageFromArray(image_array)

        prediction = predict_volume(
            self.model,
            image,
            device,
            batch_size=8
        )

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            "Prediction"
        )

        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
            volume
        )

        segmentation = segmentationNode.GetSegmentation()

        labels = np.unique(prediction)

        for label in labels:
            if label == 0:
                continue

            segmentId = segmentation.AddEmptySegment(
                f"Class_{label}",
                f"Class {label}"
            )

            binary_mask = (prediction == label).astype(np.uint8)

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                binary_mask,
                segmentationNode,
                segmentId,
                volume
            )

        segmentationNode.GetDisplayNode().SetVisibility(True)

        print("Segmentation complete!")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, features=32):
        super().__init__()

        # encoder
        self.enc1 = DoubleConv(in_channels, features)
        self.enc2 = DoubleConv(features, features * 2)
        self.enc3 = DoubleConv(features * 2, features * 4)
        self.enc4 = DoubleConv(features * 4, features * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # decoder
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features * 16, features * 8)

        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 2, features)

        # classification
        self.out = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

def predict_volume(model, image, device, batch_size=8):
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    prediction = np.zeros(volume.shape, dtype=np.uint16)

    model.eval()

    with torch.no_grad():
        for start in range(0, volume.shape[2], batch_size):
            end = min(start + batch_size, volume.shape[2])

            slices = volume[:, :, start:end]
            slices = np.moveaxis(slices, 2, 0)

            mean = slices.mean(axis=(1, 2), keepdims=True)
            std = slices.std(axis=(1, 2), keepdims=True)
            slices = (slices - mean) / (std + 1e-8)

            inputs = (
                torch.from_numpy(slices.copy())
                .float()
                .unsqueeze(1)
                .to(device)
            )

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            prediction[:, :, start:end] = np.moveaxis(
                predictions, 0, 2
            )

    return prediction
