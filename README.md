# Cell SCT
Create a machine learning model that can Segment, Classify and Track (SCT) cells.

This is a three part project, where the end goal is to have a model that can, from a series of micrographs,
- segment out the cells,
- classify them by their cell cycle phase,
- and track them over time.

To track the daily progress of this project: [journal](journal.md)

## Initial setup
- Start with the environment_setup Python Notebook. It will install the correct packages into your environment and make sure the GPU is connected.
- Right now, the segmentation part of the project is underworks but can be used to train a cell segmentation model.

## Currently working
- Environment setup is working and done with. It will be updated later on to be setup with Omero.
- The cell segmentation model training notebook is done. It can be used by anyone to understand, with little code to understand, how to train a CellPose2 model.

## Currently working on
- The evaluation Python Notebook to quantitavily and qualitatively evaluate how well a trained CellPose2 model works.
- Reading through the CellPose2 loss function (BCE.loss()) to understand how they calculate average precision.

## Currently waiting for
- Waiting for Omero to be installed in the Richmond lab computers so I can train models with that data to make them more general.
