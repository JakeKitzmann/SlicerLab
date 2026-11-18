import slicer
import qt
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget
)

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
        
