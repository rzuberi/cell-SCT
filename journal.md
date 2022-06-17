Week 1

Monday 06/06/22
- Read the CellPose2 paper
- Played around with CellPose2 with the example data

Tuesday 07/06/22
- Got cell data from Dr Hochegger's lab
- Started segmenting cells and training models in CellPose2
- Started the GitHub project

Wednesday 08/06/22
- Explored how to get the GPUs to work on the Chichester Lab computers
- Made some groundtruth data to train a model on

Thursday 09/06/22
- Tried a little more to make GPUs work
- Started a Google Colab notebook to walktrhough training a CellPose2 model with channel 1s to segment channel 0

Friday 10/06/22
- Working on Colab notebook, importing the data and labelling it


Week 2

Monday 13/06/22
- Meeting with Ivor to present what I currently have (Colab notebook), what I'm planning to have for the next week (a working segmentation model) and my issues with GPU in the lab computers
- Meeting with Helfrid to talk about what I've done with the data and the Colab segmentation notebook I'm working on
- Second meeting with Helfrid (in his office) for him to show me Omero, how a pipeline works, and some images, also asked for me to be added to the Omero VPN

Tuesday 14/06/22
- Working on Colab notebook to import a model and use it to make predictions
- Working on Colab notebook to compare predictions and grountruth, binarised the images first (not great direction, Ivor gave better things to look at such as making a bouding box around the groundtruth and putting it on the predicted mask to see if it gets the cell, pixel by pixel is the main expression he used)
- End of day quick catch-up with Ivor for him to give me advice on how to get the accuracy of the model (comparing groundtruth and predicted masks)

Wednesday 14/06/22
- Went through the protocol (with Jesus Galan, one of Helfrid's PhD students) at the Genome Centre to apply the EdU cell cycle marker to cells, left the mixture overnight and continue the next morning
- Worked on Colab notebook to single out the cells from the masks, since they should each have a unique colour I'm trying to filter out only that colour from the mask
- Quick conversation with Ivor, Mihaela and Mae about applying for PhD positions and the current interviews Ivor is conducting.

Thursday 15/06/22
- Did the final step (cleaning the cells) of the EdU staining protocol at the Genome Centre then went down to use one of the microscopes to take pictures of the cells with Jesus which he sent me the data of.
- Successfully filter out the cells and made a method to create bounding boxes around each of them (to crop the image) that can then be applied on the predictions for comparison (started a new notebook just for that part to have it be a clean demonstration).
- Successfully trained a CellPose2 model in Colab with Channel 0 images.
- Note that both of these last two points are in a messy notebook and will require cleaning-up/refactoring tomorrow. Also need to be uploaded to GitHub soon to get feedback next week. Need to learn to import from another notebook.
- Ivor's weekly reading group did not happen today.

Friday 16/06/22
- Meeting and small demonstration from Oliver Thomas (PhD student of Ivor) who showed us (Mihaela, Mae and I) how to use GitHub and creating a virtual environemt in GitHub. Tested GitHub by adding a test file to this project.
- Retried creative a virtual environment to install the GPU version of PyTorch to be able to use the GPU when training CellPose2 human-in-the-loop models, which would be much faster (failed to do this last week).
- Working on Colab notebook that demonstrates qualitative testing (with crops) of cell masks. Its added to this GitHub project.
- Added the messy Colab notebook (wlakthrough that can implement a model, train one and segment) from which the content of will be separated into smaller notebooks to the GitHub project.
