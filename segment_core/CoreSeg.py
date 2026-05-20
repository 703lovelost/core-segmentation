import logging
import os
import time
from pathlib import Path
import re
import math
from datetime import datetime

import vtk
import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin


class CoreSeg(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "CoreSeg"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Aleksey Spirkin (Novosibirsk State University)",
            "Lev Moryakin (Novosibirsk State University)",
        ]
        self.parent.helpText = (
            "Core image inference with a bundled PyTorch model. "
        )
        self.parent.acknowledgementText = (
            "This extension is based on the 3D Slicer scripted module template and was adapted "
            "for core image inference."
        )


class CoreSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.dependenciesOk = False
        self.dependencyMessage = ""

    def setup(self):
        super().setup()

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/CoreSeg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = CoreSegLogic()
        self.dependenciesOk, self.dependencyMessage = self.logic.checkDependencies(force=True)

        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.outputProbabilitySelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.outputMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.modelPathEdit.connect("currentPathChanged(QString)", self._checkCanApply)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.useBundledModelButton.connect("clicked(bool)", self.onUseBundledModel)

        self.ui.outputProbabilitySelector.baseName = "CoreSegPrediction"
        self.ui.outputProbabilitySelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]

        self.ui.outputMaskSelector.baseName = "CoreSegMask"
        self.ui.outputMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode"]

        self.ui.FinetuneSliceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._CanAddTrain)
        self.ui.FinetuneMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._CanAddTrain)
        self.ui.DatasetName.textChanged.connect(self._CanAddTrain)
        
        self.ui.FinetuneSliceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.FinetuneMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode"]

        self.ui.FinetuneSliceSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.FinetuneMaskSelector.setMRMLScene(slicer.mrmlScene)

        current = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ui.DatasetName.setPlainText(f"Dataset {current}")

        self.ui.AddTrainDataButton.connect("clicked(bool)", self.onAddData)

        self.onUseBundledModel()
        self._checkCanApply()
        self._CanAddTrain()

    def cleanup(self):
        self.removeObservers()

    def onUseBundledModel(self):
        modelPath = self.logic.defaultModelPath(self.resourcePath)
        if os.path.exists(modelPath):
            self.ui.modelPathEdit.currentPath = modelPath
        self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None):
        hasInput = self.ui.inputSelector.currentNode() is not None
        hasPrediction = self.ui.outputProbabilitySelector.currentNode() is not None
        hasModel = os.path.isfile(self.ui.modelPathEdit.currentPath)

        self.ui.applyButton.enabled = (
            self.dependenciesOk
            and hasInput
            and hasPrediction
            and hasModel
        )

        if not self.dependenciesOk:
            self.ui.applyButton.toolTip = self.dependencyMessage
            return
        if not hasInput:
            self.ui.applyButton.toolTip = "Select an input scalar volume."
            return
        if not hasPrediction:
            self.ui.applyButton.toolTip = "Create or select an output prediction volume."
            return
        if not hasModel:
            self.ui.applyButton.toolTip = "Select a valid model checkpoint file."
            return

        self.ui.applyButton.toolTip = "Run slice-wise model inference."

    def _CanAddTrain(self, caller = None, event = None):
        hasSlice = self.ui.FinetuneSliceSelector.currentNode() is not None
        hasMask = self.ui.FinetuneMaskSelector.currentNode() is not None
        normName = self._is_valid_filename(self.ui.DatasetName.toPlainText())

        self.ui.AddTrainDataButton.enabled = (
            self.dependenciesOk
            and hasSlice
            and hasMask
            and normName
        )

        if not self.dependenciesOk:
            self.ui.AddTrainDataButton.toolTip = self.dependencyMessage
            return
        if not hasSlice:
            self.ui.AddTrainDataButton.toolTip = "Select a slice scalar volume."
            return
        if not hasMask:
            self.ui.AddTrainDataButton.toolTip = "Select a mask scalar volume."
            return
        

    def _is_valid_filename(self, name):
        if not name or name.strip() == "":
            return True

        invalid = r'[<>:"/\\|?*]'

        if re.search(invalid, name):
            return False

        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }

        stem = Path(name).stem.upper()

        if stem in reserved:
            return False

        # already_existing = os.listdir(self.logic.FINETUNE_PATH)
        # if name in already_existing:
        #     return False
        
        return True

    def onApplyButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to run CoreSeg inference.", waitCursor=True):
            self.logic.process(
                inputVolume=self.ui.inputSelector.currentNode(),
                outputMaskVolume=self.ui.outputMaskSelector.currentNode(),
                outputPredictionVolume=self.ui.outputProbabilitySelector.currentNode(),
                modelPath=self.ui.modelPathEdit.currentPath,
                patch_size=self.ui.MaskResolutionBox.currentText,
                threshold=float(self.ui.thresholdSliderWidget.value),
                showResult=True,
            )

    def onAddData(self):
        with slicer.util.tryWithErrorDisplay("Failed to Add new data to train.", waitCursor=True):
            self.logic.AddData(
                SliceVolume=self.ui.FinetuneSliceSelector.currentNode(),
                MaskVolume=self.ui.FinetuneMaskSelector.currentNode(),
                DatasetName=self.ui.DatasetName.toPlainText(),
            )


class CoreSegInferenceBackend:
    DEBUG_LOGS = True
    DEBUG_PREFIX = "[CORESEG_DEBUG]"
    TARGET_SIZE = 512

    def __init__(self):
        self._torch = None
        self._A = None
        self._cv2 = None
        self._cachedModel = None
        self._cachedModelPath = None
        self._cachedDevice = None

    def _debug(self, message, *args):
        if not self.DEBUG_LOGS:
            return

        text = message % args if args else message
        text = f"{self.DEBUG_PREFIX} {text}"

        logging.info(text)

        try:
            print(text, flush=True)
            slicer.app.processEvents()
        except Exception:
            pass

    def _logArrayStats(self, name, array):
        import numpy as np
        
        if not self.DEBUG_LOGS:
            return

        x = np.asarray(array)
        if x.size == 0:
            self._debug("%s: empty array shape=%s dtype=%s", name, tuple(x.shape), x.dtype)
            return

        finiteMask = np.isfinite(x)
        finiteCount = int(finiteMask.sum())
        totalCount = int(x.size)

        if finiteCount == 0:
            self._debug("%s: shape=%s dtype=%s finite=0/%d", name, tuple(x.shape), x.dtype, totalCount)
            return

        xf = x[finiteMask].astype(np.float32, copy=False)
        self._debug(
            "%s: shape=%s dtype=%s finite=%d/%d min=%.6f max=%.6f mean=%.6f",
            name,
            tuple(x.shape),
            x.dtype,
            finiteCount,
            totalCount,
            float(xf.min()),
            float(xf.max()),
            float(xf.mean()),
        )

    def _logTensorStats(self, name, tensor):
        if not self.DEBUG_LOGS:
            return

        x = tensor.detach().cpu()
        self._debug(
            "%s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
            name,
            tuple(x.shape),
            x.dtype,
            float(x.min().item()),
            float(x.max().item()),
            float(x.mean().item()),
        )

    def _importRuntime(self):
        if self._torch is not None and self._A is not None and self._cv2 is not None:
            return self._torch, self._A, self._cv2

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not available in Slicer Python.") from exc

        try:
            import albumentations as A
        except ImportError as exc:
            raise RuntimeError("albumentations is not available in Slicer Python.") from exc

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is not available in Slicer Python.") from exc

        self._torch = torch
        self._A = A
        self._cv2 = cv2

        self._debug("Imported torch=%s", getattr(torch, "__version__", "unknown"))
        self._debug("Imported albumentations successfully")
        self._debug("Imported cv2=%s", getattr(cv2, "__version__", "unknown"))

        return self._torch, self._A, self._cv2

    def _resolveDevice(self, torchModule):
        return "cuda" if torchModule.cuda.is_available() else "cpu"

    def loadModel(self, modelPath):
        torchModule, _, _ = self._importRuntime()
        device = self._resolveDevice(torchModule)

        if (
            self._cachedModel is not None
            and self._cachedModelPath == modelPath
            and self._cachedDevice == device
        ):
            self._debug("Using cached model modelPath=%s device=%s", modelPath, device)
            return self._cachedModel, device

        self._debug("Loading model from %s", modelPath)
        model = torchModule.load(modelPath, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()

        self._cachedModel = model
        self._cachedModelPath = modelPath
        self._cachedDevice = device

        self._debug("Loaded model type=%s device=%s", type(model).__name__, device)
        return model, device

    def _preprocessSlice(self, sliceArray):
        import numpy as np

        _, A, _ = self._importRuntime()

        x = np.asarray(sliceArray, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        originalShape = tuple(x.shape)

        # x = A.Resize(self.TARGET_HEIGHT, self.TARGET_WIDTH)(image=x)["image"]
        x = A.Normalize()(image=x)["image"]
        x = np.asarray(x, dtype=np.float32)

        return x, originalShape
    
    def get_grid_positions(self, size, patch):
        n = math.ceil(size / patch)

        if n == 1:
            return [0]

        stride = (size - patch) / (n - 1)

        return [int(round(i * stride)) for i in range(n)]
        
    def predictSlice(self, sliceArray, modelPath, patch_size):
        import numpy as np

        torchModule, _, cv2 = self._importRuntime()
        model, device = self.loadModel(modelPath)

        self._logArrayStats("predictSlice/input_slice", sliceArray)

        preparedSlice, originalShape = self._preprocessSlice(sliceArray)
        self._logArrayStats("predictSlice/prepared_slice", preparedSlice)

        # inputTensor = torchModule.tensor(preparedSlice, dtype=torchModule.float32).view(
        #     1, 1, self.TARGET_HEIGHT, self.TARGET_WIDTH
        # ).to(device)
        # inputTensor = torchModule.tensor(preparedSlice, dtype=torchModule.float32)
        h, w = preparedSlice.shape

        ys = self.get_grid_positions(h, patch_size)
        xs = self.get_grid_positions(w, patch_size)
        
        patches = []
        for y in ys:
            for x in xs:
                patch = preparedSlice[y:y+patch_size, x:x+patch_size]
                patch = cv2.resize(patch, [self.TARGET_SIZE, self.TARGET_SIZE], interpolation=cv2.INTER_LINEAR)
                patches.append(patch)
        
        patches = np.stack(patches)

        patches = torchModule.as_tensor(patches, dtype=torchModule.float32, device=device).view(-1, 1, self.TARGET_SIZE, self.TARGET_SIZE)

        self._logTensorStats("predictSlice/input_patches", patches)

        with torchModule.no_grad():
            predictionTensor = torchModule.sigmoid(model(patches))

            recon = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)

            coords = [(y, x) for y in ys for x in xs]

            for i, (y, x) in enumerate(coords):
                pred = predictionTensor[i, 0].cpu().numpy().astype(np.float32)

                pred = cv2.resize(pred, [patch_size, patch_size], interpolation=cv2.INTER_LINEAR)

                recon[y:y+patch_size, x:x+patch_size] += pred
                weight[y:y+patch_size, x:x+patch_size] += 1.0
            
            reconstructed = recon / np.maximum(weight, 1.0)

            # prediction512 = predictionTensor.reshape(self.TARGET_HEIGHT, self.TARGET_WIDTH)
            # prediction512 = prediction512.detach().cpu().numpy().astype(np.float32)

        self._logArrayStats("predictSlice/reconstructed_prediction", reconstructed)

        originalHeight, originalWidth = originalShape
        predictionResized = cv2.resize(
            reconstructed,
            (int(originalWidth), int(originalHeight)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        self._logArrayStats("predictSlice/raw_prediction_resized", predictionResized)
        return predictionResized

    def predictVolume(self, volumeArray, modelPath, patch_size):
        import numpy as np
        
        array = np.asarray(volumeArray)
        self._logArrayStats("predictVolume/input_volume", array)

        if array.ndim == 2:
            prediction2d = self.predictSlice(array, modelPath, patch_size)
            self._logArrayStats("predictVolume/output_prediction_2d", prediction2d)
            return prediction2d.astype(np.float32)

        if array.ndim != 3:
            raise RuntimeError(f"Expected a 2D or 3D scalar volume, got shape {array.shape}.")

        prediction = np.zeros(array.shape, dtype=np.float32)

        for sliceIndex in range(array.shape[0]):
            self._debug("predictVolume slice=%d/%d", int(sliceIndex), int(array.shape[0] - 1))
            self._logArrayStats(f"predictVolume/input_slice_{sliceIndex}", array[sliceIndex])

            predictionSlice = self.predictSlice(array[sliceIndex], modelPath, patch_size)
            prediction[sliceIndex] = predictionSlice

            self._logArrayStats(f"predictVolume/output_prediction_slice_{sliceIndex}", predictionSlice)

        self._logArrayStats("predictVolume/output_prediction_3d", prediction)
        return prediction


class CoreSegLogic(ScriptedLoadableModuleLogic):
    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "torch": "torch",
        "albumentations": "albumentations",
        "cv2": "opencv-python",
    }

    def __init__(self):
        super().__init__()
        self.backend = CoreSegInferenceBackend()
        self._dependenciesChecked = False
        self._dependenciesOk = False
        self._dependencyMessage = ""

        base_dir = qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation)
        path = os.path.join(base_dir, "CoreSeg")
        os.makedirs(path, exist_ok=True)
        self.FINETUNE_PATH = path 


    def checkDependencies(self, force=False):
        import importlib.util

        if self._dependenciesChecked and not force:
            return self._dependenciesOk, self._dependencyMessage

        missing = []

        for importName, packageName in self.REQUIRED_PACKAGES.items():
            if importlib.util.find_spec(importName) is None:
                missing.append(packageName)

        self._dependenciesChecked = True

        if missing:
            self._dependenciesOk = False
            self._dependencyMessage = (
                "Missing Python packages in Slicer Python: "
                + ", ".join(missing)
            )
        else:
            self._dependenciesOk = True
            self._dependencyMessage = ""

        return self._dependenciesOk, self._dependencyMessage

    def requireDependencies(self):
        ok, message = self.checkDependencies()
        if not ok:
            raise RuntimeError(message)

    @staticmethod
    def defaultModelPath(resourcePathGetter):
        return resourcePathGetter("Models/default_segformer.pth")

    @staticmethod
    def _copyVolumeGeometry(referenceVolume, outputVolume):
        ijkToRas = vtk.vtkMatrix4x4()
        referenceVolume.GetIJKToRASMatrix(ijkToRas)
        outputVolume.SetIJKToRASMatrix(ijkToRas)
        outputVolume.SetOrigin(referenceVolume.GetOrigin())
        outputVolume.SetSpacing(referenceVolume.GetSpacing())
        outputVolume.CreateDefaultDisplayNodes()

    @staticmethod
    def _getOrCreateOutputSegment(segmentationNode, segmentName="CoreSegMask"):
        segmentation = segmentationNode.GetSegmentation()
        segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
        if not segmentId:
            segmentId = segmentation.AddEmptySegment(segmentName)
        return segmentId

    def process(self, inputVolume, outputMaskVolume, outputPredictionVolume, modelPath, patch_size, threshold=0.5, showResult=True):
        self.requireDependencies()
        import numpy as np
        
        if inputVolume is None:
            raise ValueError("Input volume is invalid.")
        if inputVolume.GetImageData() is None:
            raise ValueError("Input volume has no image data.")
        if outputPredictionVolume is None:
            raise ValueError("Output prediction volume is invalid.")
        if not os.path.isfile(modelPath):
            raise ValueError("Model file does not exist.")
        
        resolutionMap = {
            "1:2": 256,
            "1:4": 512,
            "1:8": 1024,
            "1:16": 2048,
        }
        patch_size = resolutionMap[patch_size]

        startTime = time.time()
        logging.info("CoreSeg inference started")

        self.backend._debug(
            "process input=%s inputClass=%s outputPrediction=%s outputPredictionClass=%s outputMask=%s threshold_unused=%.6f modelPath=%s",
            inputVolume.GetName(),
            inputVolume.GetClassName(),
            outputPredictionVolume.GetName(),
            outputPredictionVolume.GetClassName(),
            outputMaskVolume.GetName() if outputMaskVolume else "None",
            float(threshold),
            modelPath,
        )

        inputArray = np.copy(slicer.util.arrayFromVolume(inputVolume))
        self.backend._logArrayStats("process/input_array_copy", inputArray)

        predictionArray = self.backend.predictVolume(inputArray, modelPath, patch_size)
        self.backend._logArrayStats("process/prediction_array", predictionArray)

        slicer.util.updateVolumeFromArray(outputPredictionVolume, predictionArray)
        self._copyVolumeGeometry(inputVolume, outputPredictionVolume)
        outputPredictionVolume.SetName(outputPredictionVolume.GetName() or "CoreSegPrediction")
        outputPredictionVolume.CreateDefaultDisplayNodes()

        predictionDisplayNode = outputPredictionVolume.GetDisplayNode()
        if predictionDisplayNode:
            predictionDisplayNode.AutoWindowLevelOn()
            predictionDisplayNode.SetVisibility(True)

        writtenPredictionArray = slicer.util.arrayFromVolume(outputPredictionVolume)
        self.backend._logArrayStats("process/written_prediction_volume", writtenPredictionArray)

        if outputMaskVolume is not None:
            if not outputMaskVolume.IsA("vtkMRMLSegmentationNode"):
                raise ValueError("Output mask node must be a Segmentation node.")

            maskArray = (predictionArray >= float(threshold)).astype(np.uint8)
            self.backend._logArrayStats("process/mask_array", maskArray)

            outputMaskVolume.CreateDefaultDisplayNodes()
            outputMaskVolume.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

            segmentId = self._getOrCreateOutputSegment(outputMaskVolume, "CoreSegMask")
            segment = outputMaskVolume.GetSegmentation().GetSegment(segmentId)
            segment.SetName("CoreSegMask")
            segment.SetColor(1.0, 0.0, 0.0)

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                maskArray,
                outputMaskVolume,
                segmentId,
                inputVolume,
            )

            writtenMaskArray = slicer.util.arrayFromSegmentBinaryLabelmap(
                outputMaskVolume,
                segmentId,
                inputVolume,
            )
            self.backend._logArrayStats("process/written_mask_array", writtenMaskArray)

            segmentationDisplayNode = outputMaskVolume.GetDisplayNode()
            if segmentationDisplayNode:
                segmentationDisplayNode.SetVisibility(True)
                segmentationDisplayNode.SetVisibility2D(True)
                segmentationDisplayNode.SetVisibility3D(False)
                segmentationDisplayNode.SetOpacity2DFill(0.35)
                segmentationDisplayNode.SetOpacity2DOutline(1.0)

        if showResult:
            slicer.util.setSliceViewerLayers(
                background=inputVolume,
                foreground=outputPredictionVolume,
                label=None,
                fit=True,
            )

            layoutManager = slicer.app.layoutManager()
            if layoutManager is not None:
                for sliceViewName in layoutManager.sliceViewNames():
                    compositeNode = layoutManager.sliceWidget(sliceViewName).mrmlSliceCompositeNode()
                    compositeNode.SetForegroundOpacity(0.5)

        stopTime = time.time()
        logging.info(f"CoreSeg inference completed in {stopTime - startTime:.2f} seconds")
        self.backend._debug("process finished in %.2f seconds", float(stopTime - startTime))

    def AddData(self, SliceVolume, MaskVolume, DatasetName):
        self.requireDependencies()
        import numpy as np

        if SliceVolume is None:
            raise ValueError("Slice volume is invalid.")
        if SliceVolume.GetImageData() is None:
            raise ValueError("Slice volume has no image data.")
        if MaskVolume is None:
            raise ValueError("Segmented volume is invalid.")
        
        logging.info(f"Adding data to {self.FINETUNE_PATH} file name {DatasetName}")

        SliceArray = np.copy(slicer.util.arrayFromVolume(SliceVolume))

        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            MaskVolume,
            labelmapNode
        )

        MaskArray = slicer.util.arrayFromVolume(labelmapNode)

        self.backend._logArrayStats("AddData/Slices", SliceArray)
        self.backend._logArrayStats("AddData/Masks", MaskArray)

        if SliceArray.shape != MaskArray.shape:
            raise ValueError("Slices and Masks have different shapes.")

        data_path = os.path.join(self.FINETUNE_PATH, DatasetName)
        os.makedirs(data_path, exist_ok=False)

        np.save(os.path.join(data_path, "slices"), SliceArray)
        np.save(os.path.join(data_path, "masks"), MaskArray)
        



class CoreSegTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_CoreSeg_basic()

    def test_CoreSeg_basic(self):
        logic = CoreSegLogic()
        self.assertIsNotNone(logic.backend)