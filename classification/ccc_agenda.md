# Cell Cycle Classification

This agenda serves as a textual information holder for the cell cycle classification part of the project.
It should describe the current state of the project, have the TODOs, the issuses and the suggestions.

## Current state
- Nothing has been started except for this agenda.
- Helfrid sent me the notebooks to get the data from Omero.

## TODOs
- Explore the Omero data retrieving notebooks sent by Helfrid and adapt them into this project.
- Need to get ground truth data to train a CNN.
- Have a read-through the classification paper that Helfrid has sent on slack a little while ago.

## Issues
- Not sure about how to make the classes from the cell data, should figure that out first after getting Omero to work.

## Suggestions
- From the cropped nuclei/cells, augment the data
- Run a cluster on the cells' data (DAPI and EdU) to get the ground truth.
- The robustness has to be leveled into the level of the PCNA since that's what we want our CNN to classify from in the end.
- Edges of microscopy pictures might be blurry. Helfrid recommends applying a gaussian blur on the entire image and then dividing or subtracting it.
- The output from the CNN could be a confidence score (60/40, 70/30 etc.) instead of giving one class (1 or 0).
- After having classified the cells, on the original cell image (with all the cells), add textually each cell's label next to it
- The k-means clustering graph (with the dots of colors) would look great on the front of this project when showing what each section deos