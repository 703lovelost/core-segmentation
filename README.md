# Core Segmentation

Slicer 3D extension for MRI core segmentation.

## Expected input

* 2D/3D scalar volume.

## Expected output

* Prediction scalar volume.
* Segmentation mask.

## Setting up

1. Clone this repository to your `PATH_TO_SLICER/extensions` folder.
2. Install dependencies to your Slicer environment:

```
PATH_TO_SLICER/bin/PythonSlicer -m pip install -r requirements.txt
```
3. The extension should be available in 'Segmentation' module category.
