# Cell SCT
Create a machine learning model that can Segment, Classify and Track (SCT) cells.

This is a three part project, where the end goal is to have a model that can, from a series of micrographs,
- segment out the cells; 'seg'
- classify them by their cell cycle phase; 'ccc'
- and track them over time.

To track the daily progress of this project: [journal](journal.md)

## Initial setup
- Start with the environment_setup Python Notebook. It will install the correct packages into your environment and make sure the GPU is connected.
- There is also an [omero setup](omero_setup.ipynb) notebook to establish a connection if needed.
- Right now, the [segmentation part](segmentation) of the project is done and can be used to train a cell segmentation model and evaluate it qualitatively and quantitatively with comparison options.
- The classification part is under works.

## What works
- The environment and omero connection setup notebooks are done.
- The cell segmentation model training notebook is done. It can be used by anyone to understand, with little code to understand, how to train a CellPose2 model.
- The cell segmentation model evaluation notebook is done and presents the functions to qualitatively and quantitatively evaluate a model and compare different ones.
- The omero setup notebook is done and can be used to collect data as well as save data.
- The training data notebook is done and can create a training data CSV file from 4 channel images and apply the classification rules on them.

## What I'm working on
- Classifying the cells from the PCNA channel only.
