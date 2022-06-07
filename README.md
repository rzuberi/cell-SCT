# Cell SCT
Create a machine learning model that can Segment, Classify and Track (SCT) cells.

This is a three part project, where the end goal is to have a model that can, from a series of micrographs,
- segment out the cells,
- classify them by their cell cycle phase,
- and track them over time.

## Currently working on

- Exploring how CellPose2 works
- Setting up the first notebook

## Current results / what's working

- Only have the base data opened up in Fiji and in CellPose2

## Segmentation

- Creating a segmentation model from CellPose2. The idea here would be to use the cell micrographs in a human-in-the-loop training model. Once it is trained, I will put it in a notebook and upload it to this GitHub project.
- CellPose2 creators made a great notebook to implement a custom trained model: https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Z2ac5gtr-HPq
