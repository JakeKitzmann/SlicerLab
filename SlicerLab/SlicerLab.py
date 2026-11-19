import sys

import os
PACKAGE_DIR = "/nfsscratch/Users/jkitzmann/SlicerLab/slicer-packages"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_unet2d.pth")
MODEL_PATH = os.path.abspath(MODEL_PATH)

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
import sitkUtils
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

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
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

        print("Segmenting...")

        # Full geometry (spacing, origin, direction) in LPS, as SimpleITK expects
        image = sitkUtils.PullVolumeFromSlicer(volume)

        # Returned in the same voxel layout as the input volume
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

# Orientation the model was trained in: LPS, i.e. array [z, y, x] =
# [inferior->superior, anterior->posterior, right->left], axial slices along
# axis 0. This is the layout of Slicer's sample CTs (e.g. CTChest).
CANONICAL_ORIENTATION = "LPS"


def normalize_slices(x, eps=1e-8):
    """Z-score each 2D slice independently (last two axes). Matches training."""
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return (x - mean) / (std + eps)


def predict_volume(model, image, device, batch_size=8):
    """Segment a SimpleITK image of any orientation.

    The image is reoriented to CANONICAL_ORIENTATION, segmented slice by slice,
    and the prediction is reoriented back. The returned array has the same
    [k, j, i] layout as sitk.GetArrayFromImage(image).
    """
    original_orientation = (
        sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            image.GetDirection()
        )
    )
    canonical = sitk.DICOMOrient(image, CANONICAL_ORIENTATION)

    volume = sitk.GetArrayFromImage(canonical).astype(np.float32)
    prediction = np.zeros(volume.shape, dtype=np.uint8)

    model.eval()

    with torch.no_grad():
        for start in range(0, volume.shape[0], batch_size):
            end = min(start + batch_size, volume.shape[0])

            # Axial slices (array is [z, y, x])
            slices = normalize_slices(volume[start:end])

            inputs = (
                torch.from_numpy(slices.copy())
                .float()
                .unsqueeze(1)
                .to(device)
            )

            outputs = model(inputs)
            prediction[start:end] = outputs.argmax(dim=1).cpu().numpy()

    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.CopyInformation(canonical)
    prediction_image = sitk.DICOMOrient(prediction_image, original_orientation)

    return sitk.GetArrayFromImage(prediction_image)
