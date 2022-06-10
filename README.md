# Cell SCT
Create a machine learning model that can Segment, Classify and Track (SCT) cells.

This is a three part project, where the end goal is to have a model that can, from a series of micrographs,
- segment out the cells,
- classify them by their cell cycle phase,
- and track them over time.

## Problems and questions for supervisors
- Using GPU in Chichester labs requires administrator access.
- If I am able to get the GPU running, I would like to switch from Colab to DataSpell as Dr Hochegger had proposed. I rely on Colab for now since I cannot use the GPUs on the Chichester lab computers.
- I made a Colab notebook that walks through building a CellPose2 model and calculates its accuracy. It makes a mask for each of the cell microscopy pictures on channel 0 images, effectively showing where the nucleis are. What should be the end goal of segmentation? Are masks enough or should I make new images that take out the cells completely from the image? As in new cropped images of each found nucleis. 

## Currently working on

- Exploring how CellPose2 works
- Setting up the first notebook
- Colab notebook walkthrough for segmentation

## Current results / what's working

- Only have the base data opened up in Fiji and in CellPose2

## Segmentation

- Creating a segmentation model from CellPose2. The idea here would be to use the cell micrographs in a human-in-the-loop training model. Once it is trained, I will put it in a notebook and upload it to this GitHub project.
- CellPose2 creators made a great notebook to implement a custom trained model: https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Z2ac5gtr-HPq


## Using GPU in Chichester labs
- CellPose2 models take 10 minutes to train on CPU.
- The Chichester lab 2 computers have GPUs: NVIDIA Quadro M5000, with 8GB of dedicated memory.
- Using these GPUs (on Anaconda3, where CellPose2 runs) requires installing a different version of PyTorch (ref:[CellPose2 documentation](https://github.com/MouseLand/cellpose)).
This results in an error stating that the current logged-in user (me, Rehan) does not have the administrator rights to complete this installation on the lab machine:
```
EnvironmentNotWritableError: The current user does not have write permissions to the target environment.
  environment location: C:\ProgramData\Anaconda3
```

Tried (and failed) solutions:
- Using CellPose2 through Google Colab to use their free GPUs. Unfortunately, CellPose2's labelling GUI can only work through a terminal, therefore requires being run on Anaconda3 and cannot be run on a notebook (such as Colab).
- Asking Sussex IT services for administrator access. Unfortunately, after describing my problem and the error I was encountering, they informed me that giving administrator access or even helping me install that package was not a possibility.

## Using DataSpell
- Dr Hochegger proposed using DataSpell for making notebooks.
- This would depend on the status of the GPU access I have in the Chichester labs.
- DataSpell is not on the SoftwareHub (apps provided in Sussex labs).
- Using DataSpell also 

Solution
- Could you use the research money (£200) to buy a DataSpell subscription which is £29.8 for two months ([jetbrains](https://www.jetbrains.com/dataspell/buy/#commercial?billing=monthly).

## Loss function of CellPose2
