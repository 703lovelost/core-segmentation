import logging
import os
import time
import json
from pathlib import Path
import re
import math
from datetime import datetime
import sys
import ctypes


import vtk # type: ignore
import qt # type: ignore
import slicer # type: ignore
from slicer.ScriptedLoadableModule import ScriptedLoadableModule # type: ignore
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic # type: ignore
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest # type: ignore
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget # type: ignore
from slicer.util import VTKObservationMixin # type: ignore


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

        self.logic = CoreSegLogic(self.ui.TrainProgressBar)
        self.dependenciesOk, self.dependencyMessage = self.logic.checkDependencies(force=True)

        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.outputProbabilitySelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.outputMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)
        self.ui.modelPathEdit.connect("currentPathChanged(QString)", self._refreshButtons)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.useBundledModelButton.connect("clicked(bool)", self.onUseBundledModel)

        self.ui.outputProbabilitySelector.baseName = "CoreSegPrediction"
        self.ui.outputProbabilitySelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]

        self.ui.outputMaskSelector.baseName = "CoreSegMask"
        self.ui.outputMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode"]

        self.ui.FinetuneSliceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onFinetuneSliceChanged)
        self.ui.FinetuneMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._CanAddTrain)
        self.ui.DatasetName.textChanged.connect(self._CanAddTrain)
        self.ui.FinetuneDimensionEdit.textChanged.connect(self._onSubsetDimensionChanged)
        self.ui.FinetuneFromEdit.textChanged.connect(self._CanAddTrain)
        self.ui.FinetuneToEdit.textChanged.connect(self._CanAddTrain)
        
        self.ui.FinetuneSliceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.FinetuneMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode"]

        self.ui.FinetuneSliceSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.FinetuneMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.FinetuneDimensionEdit.setValidator(qt.QIntValidator(0, 2, self.ui.FinetuneDimensionEdit))
        self.ui.FinetuneFromEdit.setValidator(qt.QIntValidator(0, 2147483647, self.ui.FinetuneFromEdit))
        self.ui.FinetuneToEdit.setValidator(qt.QIntValidator(0, 2147483647, self.ui.FinetuneToEdit))

        current = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ui.DatasetName.setPlainText(f"Dataset {current}")

        self.ui.AddTrainDataButton.connect("clicked(bool)", self.onAddData)

        self.ui.MaskResolutionBox.setCurrentIndex(1)
        self.ui.LearningRateBox.setCurrentIndex(1)
        self.ui.BatchSizeBox.setCurrentIndex(2)
        self.ui.MaxEpochSpin.setValue(30)

        self.ui.StartTrainButton.connect("clicked(bool)", self.onTrainModel)

        self.onUseBundledModel()
        self._updateSubsetControls()
        self._refreshButtons()

    def _isBusy(self):
        return self.logic is not None and self.logic.isBusy()

    def cleanup(self):
        self.removeObservers()

    def onUseBundledModel(self):
        modelPath = self.logic.defaultModelPath(self.resourcePath)
        if os.path.exists(modelPath):
            self.ui.modelPathEdit.currentPath = modelPath
        self._refreshButtons()

    def _checkCanApply(self, caller=None, event=None):
        hasInput = self.ui.inputSelector.currentNode() is not None
        hasPrediction = self.ui.outputProbabilitySelector.currentNode() is not None
        hasModel = os.path.isfile(self.ui.modelPathEdit.currentPath)

        self.ui.applyButton.enabled = (
            self.dependenciesOk
            and hasInput
            and hasPrediction
            and hasModel
            and not self._isBusy()
        )

        if not self.dependenciesOk:
            self.ui.applyButton.toolTip = self.dependencyMessage
            return
        if self._isBusy():
            self.ui.applyButton.toolTip = "Another operation is running."
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

    def _onFinetuneSliceChanged(self, caller=None, event=None):
        self._updateSubsetControls(reset=True)
        self._CanAddTrain()

    def _onSubsetDimensionChanged(self, text=None):
        self._updateSubsetRangeForDimension()
        self._CanAddTrain()

    def _setSubsetControlsEnabled(self, enabled):
        enabled = bool(enabled)
        for widget in (
            self.ui.FinetuneDimensionEdit,
            self.ui.FinetuneFromEdit,
            self.ui.FinetuneToEdit,
        ):
            widget.setEnabled(enabled)
            widget.setReadOnly(not enabled)

    def _finetuneSliceArrayShape(self):
        volume = self.ui.FinetuneSliceSelector.currentNode()
        if volume is None or volume.GetImageData() is None:
            return None
        try:
            return tuple(slicer.util.arrayFromVolume(volume).shape)
        except Exception:
            return None

    def _updateSubsetControls(self, reset=False):
        shape = self._finetuneSliceArrayShape()
        has3dVolume = shape is not None and len(shape) == 3
        self._setSubsetControlsEnabled(has3dVolume)

        if not has3dVolume:
            if reset:
                self.ui.FinetuneDimensionEdit.clear()
                self.ui.FinetuneFromEdit.clear()
                self.ui.FinetuneToEdit.clear()
            return

        if reset or not self.ui.FinetuneDimensionEdit.text.strip():
            self.ui.FinetuneDimensionEdit.setText("0")
        if reset or not self.ui.FinetuneFromEdit.text.strip():
            self.ui.FinetuneFromEdit.setText("0")
        self._updateSubsetRangeForDimension(reset_to_full=reset)

    def _updateSubsetRangeForDimension(self, reset_to_full=False):
        shape = self._finetuneSliceArrayShape()
        if shape is None or len(shape) != 3:
            return

        try:
            dimension = int(self.ui.FinetuneDimensionEdit.text)
        except ValueError:
            return

        if dimension < 0 or dimension >= len(shape):
            return

        maxIndex = int(shape[dimension] - 1)
        if reset_to_full or not self.ui.FinetuneToEdit.text.strip():
            self.ui.FinetuneToEdit.setText(str(maxIndex))
            return

        try:
            currentTo = int(self.ui.FinetuneToEdit.text)
        except ValueError:
            self.ui.FinetuneToEdit.setText(str(maxIndex))
            return

        if currentTo > maxIndex:
            self.ui.FinetuneToEdit.setText(str(maxIndex))

    def _getSubsetParameters(self):
        shape = self._finetuneSliceArrayShape()
        if shape is None or len(shape) != 3:
            return None, None, None

        try:
            dimension = int(self.ui.FinetuneDimensionEdit.text)
            frameFrom = int(self.ui.FinetuneFromEdit.text)
            frameTo = int(self.ui.FinetuneToEdit.text)
        except ValueError:
            return None, None, None

        if dimension < 0 or dimension >= len(shape):
            return None, None, None
        if frameFrom < 0 or frameTo < 0:
            return None, None, None
        if frameFrom > frameTo:
            return None, None, None
        if frameTo >= shape[dimension]:
            return None, None, None

        return dimension, frameFrom, frameTo

    def _CanAddTrain(self, caller = None, event = None):
        shape = self._finetuneSliceArrayShape()
        has3dVolume = shape is not None and len(shape) == 3
        self._setSubsetControlsEnabled(has3dVolume)

        hasSlice = self.ui.FinetuneSliceSelector.currentNode() is not None
        hasMask = self.ui.FinetuneMaskSelector.currentNode() is not None
        normName = self._is_valid_filename(self.ui.DatasetName.toPlainText())
        dimension, frameFrom, frameTo = self._getSubsetParameters()
        hasValidSubset = dimension is not None and frameFrom is not None and frameTo is not None

        self.ui.AddTrainDataButton.enabled = (
            self.dependenciesOk
            and hasSlice
            and hasMask
            and normName
            and hasValidSubset
            and not self._isBusy()
        )

        if not self.dependenciesOk:
            self.ui.AddTrainDataButton.toolTip = self.dependencyMessage
            return
        if self._isBusy():
            self.ui.AddTrainDataButton.toolTip = "Another operation is running."
            return
        if not hasSlice:
            self.ui.AddTrainDataButton.toolTip = "Select a slice scalar volume."
            return
        if not hasMask:
            self.ui.AddTrainDataButton.toolTip = "Select a mask scalar volume."
            return
        if not hasValidSubset:
            self.ui.AddTrainDataButton.toolTip = "Set a valid 3D volume subset: Dimension in [0, 2], From <= To, To inside selected dimension."
            return
        if not normName:
            self.ui.AddTrainDataButton.toolTip = "Dataset name contains invalid file name characters."
            return

        self.ui.AddTrainDataButton.toolTip = "Add selected volume subset to the training dataset."

    def _CanStartTrain(self, caller=None, event=None):
        hasModel = os.path.isfile(self.ui.modelPathEdit.currentPath)

        self.ui.StartTrainButton.enabled = (
            self.dependenciesOk
            and hasModel
            and not self._isBusy()
        )

        if self._isBusy():
            self.ui.StartTrainButton.toolTip = "Another operation is running."
        elif not self.dependenciesOk:
            self.ui.StartTrainButton.toolTip = self.dependencyMessage
        elif not hasModel:
            self.ui.StartTrainButton.toolTip = "Select a valid model checkpoint file."
        else:
            self.ui.StartTrainButton.toolTip = "Start model fine-tuning."

    def _refreshButtons(self):
        self.ui.useBundledModelButton.enabled = (
            self.dependenciesOk
            and not self._isBusy()
        )

        if not self.dependenciesOk:
            self.ui.useBundledModelButton.toolTip = self.dependencyMessage
        elif self._isBusy():
            self.ui.useBundledModelButton.toolTip = "Another operation is running."
        else:
            self.ui.useBundledModelButton.toolTip = "Use bundled model checkpoint."

        self._checkCanApply()
        self._CanAddTrain()
        self._CanStartTrain()

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

    def _setBusyUi(self, busy, text=None):
        if busy:
            self.ui.applyButton.enabled = False
            self.ui.AddTrainDataButton.enabled = False
            self.ui.StartTrainButton.enabled = False
            self.ui.useBundledModelButton.enabled = False

            self.ui.applyButton.toolTip = "Another CoreSeg operation is running."
            self.ui.AddTrainDataButton.toolTip = "Another CoreSeg operation is running."
            self.ui.StartTrainButton.toolTip = "Another CoreSeg operation is running."
            self.ui.useBundledModelButton.toolTip = "Another CoreSeg operation is running."

            if text is not None:
                self.ui.StartTrainButton.text = text

            return

        self.ui.StartTrainButton.text = "Start train"
        self._refreshButtons()

    def _onInferenceFinishedUi(self):
        self._setBusyUi(False)

    def _onTrainFinishedUi(self):
        if hasattr(self.logic, "_lastTrainedModelPath") and os.path.isfile(self.logic._lastTrainedModelPath):
            self.ui.modelPathEdit.currentPath = self.logic._lastTrainedModelPath

        self._setBusyUi(False)

    def onApplyButton(self):
        self._setBusyUi(True, "Inference...")

        try:
            with slicer.util.tryWithErrorDisplay("Failed to start CoreSeg inference.", waitCursor=False):
                self.logic.RunInference(
                    inputVolume=self.ui.inputSelector.currentNode(),
                    outputMaskVolume=self.ui.outputMaskSelector.currentNode(),
                    outputPredictionVolume=self.ui.outputProbabilitySelector.currentNode(),
                    modelPath=self.ui.modelPathEdit.currentPath,
                    patch_size=self.ui.MaskResolutionBox.currentText,
                    pad_size=32,
                    threshold=float(self.ui.thresholdSliderWidget.value),
                    showResult=True,
                    finishedCallback=self._onInferenceFinishedUi,
                )
        except Exception:
            self._onInferenceFinishedUi()
            raise

    def onAddData(self):
        with slicer.util.tryWithErrorDisplay("Failed to Add new data to train.", waitCursor=True):
            dimension, frameFrom, frameTo = self._getSubsetParameters()
            self.logic.AddData(
                SliceVolume=self.ui.FinetuneSliceSelector.currentNode(),
                MaskVolume=self.ui.FinetuneMaskSelector.currentNode(),
                DatasetName=self.ui.DatasetName.toPlainText(),
                SubsampleDimension=dimension,
                SubsampleFrom=frameFrom,
                SubsampleTo=frameTo,
            )

    def onTrainModel(self):
        self._setBusyUi(True, "Training...")

        try:
            with slicer.util.tryWithErrorDisplay("Model train failed.", waitCursor=False):
                self.logic.Train(
                    modelPath=self.ui.modelPathEdit.currentPath,
                    lr=float(self.ui.LearningRateBox.currentText),
                    max_epochs=int(self.ui.MaxEpochSpin.value),
                    base_data_prop=float(self.ui.BaseDatasetProportionSlider.value),
                    batchsize=int(self.ui.BatchSizeBox.currentText),
                    val_prop=float(self.ui.ValidationProportionSlider.value),
                    finishedCallback=self._onTrainFinishedUi,
                )
        except Exception:
            self._onTrainFinishedUi()
            raise


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
    
    def _percentileNormalize(self, img, p_low=2.5, p_high=97.5):
        import numpy as np
        img = img.astype(np.float32)

        low, high = np.percentile(img, [p_low, p_high])

        img = np.clip(img, low, high)

        img = (img - low) / (high - low + 1e-8)

        img = (img * 255.0).astype(np.uint8)

        return img

    def _preprocessSlice(self, sliceArray):
        import numpy as np

        _, A, _ = self._importRuntime()

        x = np.asarray(sliceArray, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        originalShape = tuple(x.shape)

        # x = A.Resize(self.TARGET_HEIGHT, self.TARGET_WIDTH)(image=x)["image"]
        x = self._percentileNormalize(x, p_low = 0, p_high= 97.5)
        x = A.Normalize()(image=x)["image"]
        x = np.asarray(x, dtype=np.float32)

        return x, originalShape
    
    def get_grid_positions(self, size, patch):
        n = math.ceil(size / patch)

        if n == 1:
            return [0]

        stride = (size - patch) / (n - 1)

        return [int(round(i * stride)) for i in range(n)]
        
    def predictSlice(self, sliceArray, modelPath, patch_size = 512, pad_size = 32):
        import numpy as np
        torchModule, _, cv2 = self._importRuntime()
        model, device = self.loadModel(modelPath)

        self._logArrayStats("predictSlice/input_slice", sliceArray)

        preparedSlice, originalShape = self._preprocessSlice(sliceArray)
        self._logArrayStats("predictSlice/prepared_slice", preparedSlice)

        h, w = preparedSlice.shape

        if h % patch_size < pad_size * np.floor(h / patch_size) and patch_size != h:
            h_pad = pad_size
        else:
            h_pad = 0

        if w % patch_size < pad_size * np.floor(w / patch_size) and patch_size != w:
            w_pad = pad_size
        else:
            w_pad = 0
         
        ys = self.get_grid_positions(h - h_pad * 2, patch_size - h_pad * 2)
        xs = self.get_grid_positions(w - w_pad * 2, patch_size - w_pad * 2)
        
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

            reconstructed = np.zeros((h, w), dtype=np.float32)

            coords = [(y, x) for y in ys for x in xs]

            for i, (y, x) in enumerate(coords):
                pred = predictionTensor[i, 0].cpu().numpy().astype(np.float32)
                
                pred = cv2.resize(pred, [min(patch_size, h), min(patch_size, w)], interpolation=cv2.INTER_LINEAR)

                reconstructed[y:y+patch_size, x:x+patch_size] = np.maximum(reconstructed[y:y+patch_size, x:x+patch_size], pred)


        self._logArrayStats("predictSlice/reconstructed_prediction", reconstructed)

        originalHeight, originalWidth = originalShape
        predictionResized = cv2.resize(
            reconstructed,
            (int(originalWidth), int(originalHeight)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        self._logArrayStats("predictSlice/raw_prediction_resized", predictionResized)
        return predictionResized

    def predictVolume(self, volumeArray, modelPath, patch_size, pad_size, progressCallback=None):
        import numpy as np
        
        array = np.asarray(volumeArray)
        self._logArrayStats("predictVolume/input_volume", array)

        if array.ndim == 2:
            prediction2d = self.predictSlice(array, modelPath, patch_size, pad_size)
            self._logArrayStats("predictVolume/output_prediction_2d", prediction2d)

            if progressCallback is not None:
                progressCallback(1, 1)

            return prediction2d.astype(np.float32)

        if array.ndim != 3:
            raise RuntimeError(f"Expected a 2D or 3D scalar volume, got shape {array.shape}.")

        prediction = np.zeros(array.shape, dtype=np.float32)
        totalSlices = int(array.shape[0])

        for sliceIndex in range(totalSlices):
            self._debug("predictVolume slice=%d/%d", int(sliceIndex), int(totalSlices - 1))
            self._logArrayStats(f"predictVolume/input_slice_{sliceIndex}", array[sliceIndex])

            predictionSlice = self.predictSlice(array[sliceIndex], modelPath, patch_size, pad_size)
            prediction[sliceIndex] = predictionSlice

            self._logArrayStats(f"predictVolume/output_prediction_slice_{sliceIndex}", predictionSlice)

            if progressCallback is not None:
                progressCallback(sliceIndex + 1, totalSlices)

        self._logArrayStats("predictVolume/output_prediction_3d", prediction)
        return prediction


class CoreSegLogic(ScriptedLoadableModuleLogic):
    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "torch": "torch",
        "albumentations": "albumentations",
        "cv2": "opencv-python",
    }

    def __init__(self, TrainProgressBar):
        super().__init__()
        self.backend = CoreSegInferenceBackend()
        self._dependenciesChecked = False
        self._dependenciesOk = False
        self._dependencyMessage = ""

        base_dir = qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation)
        path = os.path.join(base_dir, "CoreSeg")

        self.USER_DATASET_PATH = os.path.join(path, "Datasets")
        os.makedirs(self.USER_DATASET_PATH, exist_ok=True)

        self.USER_MODEL_PATH = os.path.join(path, "Models")
        os.makedirs(self.USER_MODEL_PATH, exist_ok=True)

        self.TENSORBOARD_PATH = os.path.join(path, "Tensorboard")
        os.makedirs(self.TENSORBOARD_PATH, exist_ok=True)

        moduleDir = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DATASET_PATH = os.path.join(moduleDir, 'Resources', 'Finetune')

        if not os.path.exists(self.BASE_DATASET_PATH):
            raise ValueError(f'Base dataset path {self.BASE_DATASET_PATH} doest not exist')
        
        self.trainProcess = None
        self.inferenceProcess = None
        self.TrainProgressBar = TrainProgressBar

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

    def isProcessRunning(self, process):
        return process is not None and process.state() != qt.QProcess.NotRunning

    def isTrainRunning(self):
        return self.isProcessRunning(self.trainProcess)

    def isInferenceRunning(self):
        return self.isProcessRunning(self.inferenceProcess)

    def isBusy(self):
        return self.isTrainRunning() or self.isInferenceRunning()

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

    def process(self, inputVolume, outputMaskVolume, outputPredictionVolume, modelPath, patch_size, pad_size, threshold=0.5, showResult=True):
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

        predictionArray = self.backend.predictVolume(inputArray, modelPath, patch_size, pad_size)
        self.backend._logArrayStats("process/prediction_array", predictionArray)

        self._applyInferenceResult(
            predictionArray=predictionArray,
            inputVolume=inputVolume,
            outputMaskVolume=outputMaskVolume,
            outputPredictionVolume=outputPredictionVolume,
            threshold=threshold,
            showResult=showResult,
        )

        stopTime = time.time()
        logging.info(f"CoreSeg inference completed in {stopTime - startTime:.2f} seconds")
        self.backend._debug("process finished in %.2f seconds", float(stopTime - startTime))

    def _applyInferenceResult(
        self,
        predictionArray,
        inputVolume,
        outputMaskVolume,
        outputPredictionVolume,
        threshold=0.5,
        showResult=True,
    ):
        import numpy as np

        slicer.util.updateVolumeFromArray(outputPredictionVolume, predictionArray)
        self._copyVolumeGeometry(inputVolume, outputPredictionVolume)

        outputPredictionVolume.SetName(
            outputPredictionVolume.GetName() or "CoreSegPrediction"
        )
        outputPredictionVolume.CreateDefaultDisplayNodes()

        predictionDisplayNode = outputPredictionVolume.GetDisplayNode()
        if predictionDisplayNode:
            predictionDisplayNode.AutoWindowLevelOn()
            predictionDisplayNode.SetVisibility(True)

        if outputMaskVolume is not None:
            if not outputMaskVolume.IsA("vtkMRMLSegmentationNode"):
                raise ValueError("Output mask node must be a Segmentation node.")

            maskArray = (predictionArray >= float(threshold)).astype(np.uint8)

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
                    compositeNode = layoutManager.sliceWidget(
                        sliceViewName
                    ).mrmlSliceCompositeNode()
                    compositeNode.SetForegroundOpacity(0.5)

    @staticmethod
    def _matrixToList(matrix):
        return [
            [float(matrix.GetElement(row, column)) for column in range(4)]
            for row in range(4)
        ]

    @staticmethod
    def _safeSaveNode(node, filePath):
        if not slicer.util.saveNode(node, filePath):
            raise RuntimeError(f"Failed to save node to {filePath}")

    def _writeDatasetMetadata(self, dataPath, numpyPath, nrrdPath, SliceVolume, MaskVolume, labelmapNode, SliceArray, MaskArray, SubsampleDimension, SubsampleFrom, SubsampleTo):
        sourceIjkToRas = vtk.vtkMatrix4x4()
        SliceVolume.GetIJKToRASMatrix(sourceIjkToRas)

        offset = self._arrayAxisToIjkOffset(SubsampleDimension, SubsampleFrom)
        originPoint = [float(offset[0]), float(offset[1]), float(offset[2]), 1.0]
        rasPoint = [0.0, 0.0, 0.0, 1.0]
        sourceIjkToRas.MultiplyPoint(originPoint, rasPoint)

        subsetIjkToRas = vtk.vtkMatrix4x4()
        subsetIjkToRas.DeepCopy(sourceIjkToRas)
        subsetIjkToRas.SetElement(0, 3, rasPoint[0])
        subsetIjkToRas.SetElement(1, 3, rasPoint[1])
        subsetIjkToRas.SetElement(2, 3, rasPoint[2])

        metadata = {
            "format_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "array_order": "zyx",
            "subset": {
                "dimension": int(SubsampleDimension),
                "from": int(SubsampleFrom),
                "to": int(SubsampleTo),
                "inclusive": True,
            },
            "image": {
                "node_name": SliceVolume.GetName(),
                "class_name": SliceVolume.GetClassName(),
                "shape": list(SliceArray.shape),
                "dtype": str(SliceArray.dtype),
                "spacing": [float(value) for value in SliceVolume.GetSpacing()],
                "origin": [float(rasPoint[0]), float(rasPoint[1]), float(rasPoint[2])],
                "ijk_to_ras": self._matrixToList(subsetIjkToRas),
                "numpy_path": os.path.relpath(os.path.join(numpyPath, "slices.npy"), dataPath),
                "nrrd_path": os.path.relpath(os.path.join(nrrdPath, "slices.nrrd"), dataPath),
            },
            "mask": {
                "node_name": MaskVolume.GetName(),
                "class_name": MaskVolume.GetClassName(),
                "shape": list(MaskArray.shape),
                "dtype": str(MaskArray.dtype),
                "numpy_path": os.path.relpath(os.path.join(numpyPath, "masks.npy"), dataPath),
                "segmentation_nrrd_path": os.path.relpath(os.path.join(nrrdPath, "masks.seg.nrrd"), dataPath),
            },
        }

        with open(os.path.join(dataPath, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _subsetArray(array, dimension, frameFrom, frameTo):
        selection = [slice(None)] * array.ndim
        selection[int(dimension)] = slice(int(frameFrom), int(frameTo) + 1)
        return array[tuple(selection)]

    @staticmethod
    def _arrayAxisToIjkOffset(dimension, frameFrom):
        offset = [0, 0, 0]
        offset[2 - int(dimension)] = int(frameFrom)
        return offset

    @staticmethod
    def _copySubsetVolumeGeometry(referenceVolume, outputVolume, dimension, frameFrom):
        ijkToRas = vtk.vtkMatrix4x4()
        referenceVolume.GetIJKToRASMatrix(ijkToRas)

        offset = CoreSegLogic._arrayAxisToIjkOffset(dimension, frameFrom)
        originPoint = [float(offset[0]), float(offset[1]), float(offset[2]), 1.0]
        rasPoint = [0.0, 0.0, 0.0, 1.0]
        ijkToRas.MultiplyPoint(originPoint, rasPoint)

        subsetIjkToRas = vtk.vtkMatrix4x4()
        subsetIjkToRas.DeepCopy(ijkToRas)
        subsetIjkToRas.SetElement(0, 3, rasPoint[0])
        subsetIjkToRas.SetElement(1, 3, rasPoint[1])
        subsetIjkToRas.SetElement(2, 3, rasPoint[2])

        outputVolume.SetIJKToRASMatrix(subsetIjkToRas)
        outputVolume.SetOrigin(rasPoint[0], rasPoint[1], rasPoint[2])
        outputVolume.SetSpacing(referenceVolume.GetSpacing())
        outputVolume.CreateDefaultDisplayNodes()

    @staticmethod
    def _toShortPath(path):
            path = os.path.abspath(path)


            if os.name != "nt":
                return path

            buffer = ctypes.create_unicode_buffer(4096)
            result = ctypes.windll.kernel32.GetShortPathNameW(
                path,
                buffer,
                4096
            )
            if result == 0:
                return path
            return buffer.value

    def _createSubsetScalarVolumeNode(self, sourceVolume, array, name, dimension, frameFrom):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        slicer.util.updateVolumeFromArray(node, array)
        self._copySubsetVolumeGeometry(sourceVolume, node, dimension, frameFrom)
        return node

    def _createSubsetLabelMapNode(self, sourceVolume, array, name, dimension, frameFrom):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        slicer.util.updateVolumeFromArray(node, array)
        self._copySubsetVolumeGeometry(sourceVolume, node, dimension, frameFrom)
        return node

    def _onInferenceStdout(self):
        data = self.inferenceProcess.readAllStandardOutput()
        text = data.data().decode("utf-8", errors="ignore")

        logging.info(f"INFERENCE NODE: {text}")

        for line in text.splitlines():
            line = line.strip()

            if line.startswith("PROGRESS"):
                try:
                    _, current, total = line.split(":")

                    current = int(current)
                    total = int(total)

                    percent = int((current / total) * 100)

                    self.TrainProgressBar.setValue(percent)
                    self.TrainProgressBar.setFormat(
                        f"Inference {current}/{total} slices (%p%)"
                    )
                except Exception as e:
                    print("Inference progress parse error:", e)
    
    def _onInferenceStderr(self):
        data = self.inferenceProcess.readAllStandardError()
        text = data.data().decode("utf-8", errors="ignore")
        logging.error(text)

    def _onInferenceFinished(self, exitCode):
        try:
            if exitCode != 0:
                slicer.util.errorDisplay(
                    f"Inference failed\nExit code: {exitCode}"
                )
                return

            import numpy as np

            predictionArray = np.load(self._inferenceOutputPath)

            self._applyInferenceResult(
                predictionArray=predictionArray,
                inputVolume=self._inferenceInputVolume,
                outputMaskVolume=self._inferenceOutputMaskVolume,
                outputPredictionVolume=self._inferenceOutputPredictionVolume,
                threshold=self._inferenceThreshold,
                showResult=self._inferenceShowResult,
            )

            slicer.util.infoDisplay("Inference completed")

        finally:
            self.inferenceProcess = None

            if hasattr(self, "_inferenceFinishedCallback") and self._inferenceFinishedCallback:
                self._inferenceFinishedCallback()

    def _onTrainStdout(self):
        data = self.trainProcess.readAllStandardOutput()
        text = data.data().decode("utf-8", errors="ignore")
        logging.info(f'TRAIN NODE: {text}')

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("PROGRESS"):
                try:
                    _, epoch, max_epochs = line.split(":")

                    epoch = int(epoch)
                    max_epochs = int(max_epochs)

                    percent = int((epoch / max_epochs) * 100)

                    self.TrainProgressBar.setValue(percent)
                    self.TrainProgressBar.setFormat(
                        f"{epoch}/{max_epochs} epochs (%p%)"
                    )
                except Exception as e:
                    print("Progress parse error:", e)
                    
    def _onTrainFinished(self, exitCode):
        try:
            if exitCode == 0:
                slicer.util.infoDisplay("Training completed")
            else:
                slicer.util.errorDisplay(
                    f"Training failed\nExit code: {exitCode}"
                )
        finally:
            self.trainProcess = None

            if hasattr(self, "_trainFinishedCallback") and self._trainFinishedCallback:
                self._trainFinishedCallback()

    def _onTrainStderr(self):
        data = self.trainProcess.readAllStandardError()
        text = data.data().decode("utf-8", errors="ignore")
        logging.error(text)

    @staticmethod
    def _pythonSlicerExecutable():
        exeName = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        return os.path.join(slicer.app.slicerHome, "bin", exeName)

    def RunInference(
        self,
        inputVolume,
        outputMaskVolume,
        outputPredictionVolume,
        modelPath,
        patch_size,
        pad_size,
        threshold=0.5,
        showResult=True,
        finishedCallback=None,
    ):
        self.requireDependencies()

        if self.isBusy():
            raise RuntimeError("Another CoreSeg operation is already running.")

        import numpy as np
        import tempfile

        if inputVolume is None:
            raise ValueError("Input volume is invalid.")
        if inputVolume.GetImageData() is None:
            raise ValueError("Input volume has no image data.")
        if outputPredictionVolume is None:
            raise ValueError("Output prediction volume is invalid.")
        if not os.path.isfile(modelPath):
            raise ValueError("Model file does not exist.")

        self._inferenceFinishedCallback = finishedCallback
        self._inferenceInputVolume = inputVolume
        self._inferenceOutputMaskVolume = outputMaskVolume
        self._inferenceOutputPredictionVolume = outputPredictionVolume
        self._inferenceThreshold = float(threshold)
        self._inferenceShowResult = bool(showResult)

        workDir = tempfile.mkdtemp(prefix="coreseg_inference_")
        inputPath = os.path.join(workDir, "input.npy")
        outputPath = os.path.join(workDir, "prediction.npy")

        inputArray = np.copy(slicer.util.arrayFromVolume(inputVolume))
        np.save(inputPath, inputArray)

        self._inferenceWorkDir = workDir
        self._inferenceInputPath = inputPath
        self._inferenceOutputPath = outputPath

        moduleDir = os.path.dirname(os.path.abspath(__file__))

        inferenceScript = os.path.join(
            moduleDir,
            "Resources",
            "inference.py",
        )

        pythonExecutable = self._pythonSlicerExecutable()

        logging.info(f"PYTHON: {pythonExecutable}")
        logging.info(f"Inference Script: {inferenceScript}")

        args = [
            self._toShortPath(inferenceScript),
            "--input_path", self._toShortPath(inputPath),
            "--output_path", self._toShortPath(outputPath),
            "--model_path", self._toShortPath(modelPath),
            "--patch_size", str(patch_size),
            "--pad_size", str(pad_size),
        ]

        self.TrainProgressBar.setValue(0)
        self.TrainProgressBar.setFormat("Inference 0/%m slices (%p%)")

        self.inferenceProcess = qt.QProcess()

        self.inferenceProcess.readyReadStandardOutput.connect(
            self._onInferenceStdout
        )

        self.inferenceProcess.readyReadStandardError.connect(
            self._onInferenceStderr
        )

        self.inferenceProcess.finished.connect(
            self._onInferenceFinished
        )

        self.inferenceProcess.start(
            pythonExecutable,
            args,
        )

    def AddData(
        self, 
        SliceVolume, 
        MaskVolume, 
        DatasetName, 
        SubsampleDimension=None, 
        SubsampleFrom=None, 
        SubsampleTo=None
    ):
        self.requireDependencies()
        import numpy as np

        if SliceVolume is None:
            raise ValueError("Slice volume is invalid.")
        if SliceVolume.GetImageData() is None:
            raise ValueError("Slice volume has no image data.")
        if MaskVolume is None:
            raise ValueError("Segmented volume is invalid.")
        
        logging.info(f"Adding data to {self.USER_DATASET_PATH} file name {DatasetName}")

        FullSliceArray = np.copy(slicer.util.arrayFromVolume(SliceVolume))
        if FullSliceArray.ndim != 3:
            raise ValueError(f"Raw volume must be a 3D tensor, got shape {FullSliceArray.shape}.")

        if SubsampleDimension is None or SubsampleFrom is None or SubsampleTo is None:
            raise ValueError("Subset parameters are required.")

        SubsampleDimension = int(SubsampleDimension)
        SubsampleFrom = int(SubsampleFrom)
        SubsampleTo = int(SubsampleTo)

        if SubsampleDimension < 0 or SubsampleDimension >= FullSliceArray.ndim:
            raise ValueError("Subset dimension must be 0, 1, or 2.")
        if SubsampleFrom < 0 or SubsampleTo < SubsampleFrom or SubsampleTo >= FullSliceArray.shape[SubsampleDimension]:
            raise ValueError("Subset range is outside selected raw volume dimension.")

        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            MaskVolume,
            labelmapNode
        )

        FullMaskArray = np.copy(slicer.util.arrayFromVolume(labelmapNode))

        self.backend._logArrayStats("AddData/FullSlices", FullSliceArray)
        self.backend._logArrayStats("AddData/FullMasks", FullMaskArray)

        if FullSliceArray.shape != FullMaskArray.shape:
            raise ValueError("Slices and Masks have different shapes.")

        SliceArray = np.copy(self._subsetArray(FullSliceArray, SubsampleDimension, SubsampleFrom, SubsampleTo))
        MaskArray = np.copy(self._subsetArray(FullMaskArray, SubsampleDimension, SubsampleFrom, SubsampleTo))

        self.backend._logArrayStats("AddData/Slices", SliceArray)
        self.backend._logArrayStats("AddData/Masks", MaskArray)
        
        data_path = os.path.join(self.USER_DATASET_PATH, DatasetName)
        numpy_path = os.path.join(data_path, "numpy")
        nrrd_path = os.path.join(data_path, "nrrd")

        os.makedirs(data_path, exist_ok=False)
        os.makedirs(numpy_path, exist_ok=False)
        os.makedirs(nrrd_path, exist_ok=False)

        np.save(os.path.join(numpy_path, "slices.npy"), SliceArray)
        np.save(os.path.join(numpy_path, "masks.npy"), MaskArray)

        subsetSliceNode = None
        subsetLabelmapNode = None
        subsetSegmentationNode = None
        try:
            subsetSliceNode = self._createSubsetScalarVolumeNode(
                SliceVolume,
                SliceArray,
                "CoreSegSubsetSlices",
                SubsampleDimension,
                SubsampleFrom,
            )
            subsetLabelmapNode = self._createSubsetLabelMapNode(
                SliceVolume,
                MaskArray,
                "CoreSegSubsetMasksLabelmap",
                SubsampleDimension,
                SubsampleFrom,
            )
            subsetSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "CoreSegSubsetMasks")
            subsetSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(subsetSliceNode)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                subsetLabelmapNode,
                subsetSegmentationNode,
            )

            self._safeSaveNode(subsetSliceNode, os.path.join(nrrd_path, "slices.nrrd"))
            self._safeSaveNode(subsetSegmentationNode, os.path.join(nrrd_path, "masks.seg.nrrd"))
        finally:
            for node in (subsetSegmentationNode, subsetLabelmapNode, subsetSliceNode, labelmapNode):
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)

        self._writeDatasetMetadata(
            data_path,
            numpy_path,
            nrrd_path,
            SliceVolume,
            MaskVolume,
            labelmapNode,
            SliceArray,
            MaskArray,
            SubsampleDimension,
            SubsampleFrom,
            SubsampleTo,
        )
    
    def Train(
        self, 
        modelPath, 
        lr, 
        max_epochs, 
        base_data_prop, 
        batchsize, 
        val_prop, 
        finishedCallback=None
    ):
        self.requireDependencies()

        if self.isBusy():
            raise RuntimeError("Another CoreSeg operation is already running.")

        self._trainFinishedCallback = finishedCallback

        moduleDir = os.path.dirname(os.path.abspath(__file__))

        trainerScript = os.path.join(
            moduleDir, 
            "Resources",
            "finetune.py",
        )

        pythonExecutable = self._pythonSlicerExecutable()
        logging.info(f"PYTHON: {pythonExecutable}")

        trainerScript = self._toShortPath(trainerScript)
        logging.info(f"Trainer Script: {trainerScript}")

        args = [
            trainerScript,
            "--model_path", self._toShortPath(modelPath),
            "--lr", str(lr),
            "--max_epochs", str(max_epochs),
            "--base_data_path", self._toShortPath(self.BASE_DATASET_PATH),
            "--user_data_path", self._toShortPath(self.USER_DATASET_PATH),
            "--base_prop",  str(base_data_prop),
            "--val_prop", str(val_prop),
            "--batchsize", str(batchsize),
            "--output_model_path", self._toShortPath(self.USER_MODEL_PATH),
            "--tensorboard_path", self._toShortPath(self.TENSORBOARD_PATH),
        ]

        self.trainProcess = qt.QProcess()

        self.trainProcess.readyReadStandardOutput.connect(
            self._onTrainStdout
        )

        self.trainProcess.finished.connect(
            self._onTrainFinished
        )

        self.trainProcess.readyReadStandardError.connect(
            self._onTrainStderr
        )

        self.trainProcess.start(
            pythonExecutable,
            args
        )

        self.tensorboardProcess = qt.QProcess()

        self.tensorboardProcess.readyReadStandardOutput.connect(
            self.on_out
        )

        self.tensorboardProcess.readyReadStandardError.connect(
            self.on_error
        )

        host = "localhost"
        port = get_free_port(host)

        args = [
            "--logdir", self.TENSORBOARD_PATH,
            "--port", str(port),
            "--host", host
        ]

        self.tensorboardProcess.start("tensorboard", args)

        logging.info(f'tensorboard running at {host+':'+str(port)}')
    
    def on_error(self):
        data = self.tensorboardProcess.readAllStandardError()
        text = data.data().decode("utf-8", errors="ignore")
        logging.error(f'TENSORBOARD NODE: {text}')

    def on_out(self):
        data = self.tensorboardProcess.readAllStandardOutput()
        text = data.data().decode("utf-8", errors="ignore")
        logging.info(f'TENSORBOARD NODE: {text}')

import socket

def get_free_port(host):
    s = socket.socket()
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port 


class CoreSegTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_CoreSeg_basic()

    def test_CoreSeg_basic(self):
        logic = CoreSegLogic()
        self.assertIsNotNone(logic.backend)