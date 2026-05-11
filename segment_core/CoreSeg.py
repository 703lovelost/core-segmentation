import logging
import os
import time

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

        self.onUseBundledModel()
        self.ui.modelInfoLabel.text = self.logic.defaultModelDescription()
        self._checkCanApply()

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

    def onApplyButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to run CoreSeg inference.", waitCursor=True):
            self.logic.process(
                inputVolume=self.ui.inputSelector.currentNode(),
                outputMaskVolume=self.ui.outputMaskSelector.currentNode(),
                outputPredictionVolume=self.ui.outputProbabilitySelector.currentNode(),
                modelPath=self.ui.modelPathEdit.currentPath,
                threshold=float(self.ui.thresholdSliderWidget.value),
                showResult=True,
            )


class CoreSegInferenceBackend:
    DEBUG_LOGS = True
    DEBUG_PREFIX = "[CORESEG_DEBUG]"
    TARGET_HEIGHT = 512
    TARGET_WIDTH = 512

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

        x = A.Resize(self.TARGET_HEIGHT, self.TARGET_WIDTH)(image=x)["image"]
        x = A.Normalize()(image=x)["image"]
        x = np.asarray(x, dtype=np.float32)

        return x, originalShape

    def predictSlice(self, sliceArray, modelPath):
        import numpy as np

        torchModule, _, cv2 = self._importRuntime()
        model, device = self.loadModel(modelPath)

        self._logArrayStats("predictSlice/input_slice", sliceArray)

        preparedSlice, originalShape = self._preprocessSlice(sliceArray)
        self._logArrayStats("predictSlice/prepared_slice", preparedSlice)

        inputTensor = torchModule.tensor(preparedSlice, dtype=torchModule.float32).view(
            1, 1, self.TARGET_HEIGHT, self.TARGET_WIDTH
        ).to(device)
        self._logTensorStats("predictSlice/input_tensor", inputTensor)

        with torchModule.no_grad():
            predictionTensor = model(inputTensor)
            self._logTensorStats("predictSlice/raw_model_output", predictionTensor)

            prediction512 = predictionTensor.reshape(self.TARGET_HEIGHT, self.TARGET_WIDTH)
            prediction512 = prediction512.detach().cpu().numpy().astype(np.float32)

        self._logArrayStats("predictSlice/raw_prediction_512", prediction512)

        originalHeight, originalWidth = originalShape
        predictionResized = cv2.resize(
            prediction512,
            (int(originalWidth), int(originalHeight)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        self._logArrayStats("predictSlice/raw_prediction_resized", predictionResized)
        return predictionResized

    def predictVolume(self, volumeArray, modelPath):
        array = np.asarray(volumeArray)
        self._logArrayStats("predictVolume/input_volume", array)

        if array.ndim == 2:
            prediction2d = self.predictSlice(array, modelPath)
            self._logArrayStats("predictVolume/output_prediction_2d", prediction2d)
            return prediction2d.astype(np.float32)

        if array.ndim != 3:
            raise RuntimeError(f"Expected a 2D or 3D scalar volume, got shape {array.shape}.")

        prediction = np.zeros(array.shape, dtype=np.float32)

        for sliceIndex in range(array.shape[0]):
            self._debug("predictVolume slice=%d/%d", int(sliceIndex), int(array.shape[0] - 1))
            self._logArrayStats(f"predictVolume/input_slice_{sliceIndex}", array[sliceIndex])

            predictionSlice = self.predictSlice(array[sliceIndex], modelPath)
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
    def defaultModelDescription():
        return (
            "Model description."
        )
        # return (
        #     "Each slice is resized to 512x512, normalized with albumentations.Normalize(), "
        #     "passed to the model as a [1, 1, 512, 512] tensor, and the raw prediction map is "
        #     "reshaped back to 512x512 and resized to the original slice size. "
        #     "Binary mask is produced by thresholding the prediction volume."
        # )

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

    def process(self, inputVolume, outputMaskVolume, outputPredictionVolume, modelPath, threshold=0.5, showResult=True):
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

        predictionArray = self.backend.predictVolume(inputArray, modelPath)
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


class CoreSegTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_CoreSeg_basic()

    def test_CoreSeg_basic(self):
        logic = CoreSegLogic()
        self.assertTrue(isinstance(logic.defaultModelDescription(), str))