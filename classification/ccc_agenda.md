# Cell Cycle Classification

This agenda serves as a textual information holder for the cell cycle classification part of the project.
It should describe the current state of the project, have the TODOs, the issuses and the suggestions.

## Current state
- Tried out a lot of different models.
- The best is a lenet from scratch CNN that gets a 67% accuracy.
- Otherwise the most reliable is 3 binary classifiers (e.g. G1 or not) with linear regression.

## TODOs
- Change the training data to not be crops but only the segmentation mask. Many reasons for why that is better but mainly: the background adds noise, we need to substract it.
- I made a document with all of the papers that dealt with a similar task, I should use some of their ideas/models.

## Issues
- Struggling to implement a better model. The ones that collect features from cells seem to use obscure techniques that I cannot find code source of.

## Suggestions
- From the cropped nuclei/cells, augment the data
- Run a cluster on the cells' data (DAPI and EdU) to get the ground truth.
- The robustness has to be leveled into the level of the PCNA since that's what we want our CNN to classify from in the end.
- Edges of microscopy pictures might be blurry. Helfrid recommends applying a gaussian blur on the entire image and then dividing or subtracting it.
- The output from the CNN could be a confidence score (60/40, 70/30 etc.) instead of giving one class (1 or 0).
- After having classified the cells, on the original cell image (with all the cells), add textually each cell's label next to it
- The k-means clustering graph (with the dots of colors) would look great on the front of this project when showing what each section does.
