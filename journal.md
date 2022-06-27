# JRA Project Journal (Diary)

## Week 1

### Monday 06/06/22
- Read the CellPose2 paper.
- Played around with CellPose2 with the example data.

### Tuesday 07/06/22
- Got cell data from Dr Hochegger's lab.
- Started segmenting cells and training models in CellPose2.
- Started the GitHub project.

### Wednesday 08/06/22
- Explored how to get the GPUs to work on the Chichester Lab computers.
- Made some groundtruth data to train a model on.

### Thursday 09/06/22
- Tried a little more to make GPUs work.
- Started a Google Colab notebook to walktrhough training a CellPose2 model with channel 1s to segment channel 0.

### Friday 10/06/22
- Working on Colab notebook, importing the data and labelling it.


## Week 2

### Monday 13/06/22
- Meeting with Ivor to present what I currently have (Colab notebook), what I'm planning to have for the next week (a working segmentation model) and my issues with GPU in the lab computers.
- Meeting with Helfrid to talk about what I've done with the data and the Colab segmentation notebook I'm working on.
- Second meeting with Helfrid (in his office) for him to show me Omero, how a pipeline works, and some images, also asked for me to be added to the Omero VPN.

### Tuesday 14/06/22
- Working on Colab notebook to import a model and use it to make predictions.
- Working on Colab notebook to compare predictions and grountruth, binarised the images first (not great direction, Ivor gave better things to look at such as making a bouding box around the groundtruth and putting it on the predicted mask to see if it gets the cell, pixel by pixel is the main expression he used).
- End of day quick catch-up with Ivor for him to give me advice on how to get the accuracy of the model (comparing groundtruth and predicted masks).

### Wednesday 14/06/22
- Went through the protocol (with Jesus Galan, one of Helfrid's PhD students) at the Genome Centre to apply the EdU cell cycle marker to cells, left the mixture overnight and continue the next morning.
- Worked on Colab notebook to single out the cells from the masks, since they should each have a unique colour I'm trying to filter out only that colour from the mask.
- Quick conversation with Ivor, Mihaela and Mae about applying for PhD positions and the current interviews Ivor is conducting.

### Thursday 15/06/22
- Did the final step (cleaning the cells) of the EdU staining protocol at the Genome Centre then went down to use one of the microscopes to take pictures of the cells with Jesus which he sent me the data of.
- Successfully filter out the cells and made a method to create bounding boxes around each of them (to crop the image) that can then be applied on the predictions for comparison (started a new notebook just for that part to have it be a clean demonstration).
- Successfully trained a CellPose2 model in Colab with Channel 0 images.
- Note that both of these last two points are in a messy notebook and will require cleaning-up/refactoring tomorrow. Also need to be uploaded to GitHub soon to get feedback next week. Need to learn to import from another notebook.
- Ivor's weekly reading group did not happen today.

### Friday 16/06/22
- Meeting and small demonstration from Oliver Thomas (PhD student of Ivor) who showed us (Mihaela, Mae and I) how to use GitHub and creating a virtual environemt in GitHub. Tested GitHub by adding a test file to this project.
- Retried creative a virtual environment to install the GPU version of PyTorch to be able to use the GPU when training CellPose2 human-in-the-loop models, which would be much faster (failed to do this last week). I've been able to install the right version of PyTorch on a virtual environement (that's a win) but unable to make CellPose2 us it (for now). CellPose2 says "[INFO] TORCH CUDA version not installed/working.".
- Working on Colab notebook that demonstrates qualitative testing (with crops) of cell masks. Its added to this GitHub project.
- Added the messy Colab notebook (wlakthrough that can implement a model, train one and segment) from which the content of will be separated into smaller notebooks to the GitHub project.


## Week 3

### Monday 20/06/22
- Got GPU working for CellPose2 in virtual environment, much faster, very happy.
- Activated education DataSpell license to use it for notebooks instead of Colab.
- Trying to make DataSpell, which uses Anaconda, install the PyTorch GPU to actually use the GPU through the DataSpell notebook. Currently not working. It seems to create the notebook in the virtual environement, but when exectuing conda commands from the notebook (not the terminal), it runs in the "base" environement, the one in which I do not have the permissions to install the PyTorch GPU which is needed to run CellPose2 faster.
- Weekly meeting with only Helfrid as Ivor is away. I caught him up on my progress and talked a little bit about the experience in his lab with Jesus. Shared the GitHub with him (therefore, making it urgent to clean up).

### Tuesday 21/06/22
- Helfried notified me that I should have access to Omero now. With GlobalProtect activated on his computer, with my credentials, he can connect to Omero. I cannot from the lab machine becasue I do not have GlobalProtect, which would connect me to the internal VPN. An email was sent and Helfrid put the research computing team (that knows about Omero) in contact with Christopher Sothcott who manages the Chichester lab computers.
- Jupyter refused to work (apparently not in the same file) when launched today from DataSpell. I created a new directory in DataSpell and created a Jupyter notebook in there. It was correctly put in the 'workspace' conda environement, and it was connected with the GPU. Bascially, yesterday's DataSpell issues seemed to have been solved by creating a new folder in a different directory. I'm still debating how to make these notebooks to make sure they're useful. I'm thinking of making one to just train a CellPose model and just tell the user to put the training and testing data in the right folder immediatly, and use the same to segment it.
- While the research computing team and ITS team are sorting out the Omero issue, I am working on the skeleton of the DataSpell notebooks that will directly connect to Omero. Right now, I am trying to make the notebook, from a new conda environment, install the correct PyTorch and CellPose packages and connect to the GPU (and test if its all working). Then I will look into how to connect a Python notebook to Omero to get the data from there.

### Wednesday 22/06/22
- The back-and-forth emails for GlobalProtect and Omero seem to have concluded with Luke Igerson looking into installing GlobalProtect into the Chichester Lab PCs.
- I felt a little lost yesterday, not knowing how to get these notebooks going with the correct environment and packages. The solution is a requirements.txt file in our GitHub project. That file then needs to be with the ```pip install -r requirements.txt``` command.
- I spent some time trying to understand how to, from the requirements file, install the GPU activated PyTorch (needing the correct CudaToolkit version installed) in our environment. It now works and is connected to the GPU, but the environment in which the notebook is run must be in the ".conda" folder not the "Anaconda3" one.
- I made the "testing_jup.ipynb" file (for which I'll change the name later) that walks through getting in the correct virtual environment, installing the packages from the requirements.txt file, checking that PyTorch is correctly installed, and checking that PyTorch and CellPose both have access to the GPU.

### Thursday 23/06/22
- Luke Ingerson will install GlobalProtect in two computers in the Richmond 4B9 lab. GlobalProtect will be installed remotely. It will be in the Richmond lab as the Chichester labs have ongoing renovations, so as to not do the work twice. The computers in the Richmond labs have Nvidia Quadron P1000 4GB GPUs. This should be fine for model training, but if it isn't I will ask Luke Ingerson if he can install GlobalProtect in one of the computers in the Chichester labs after they have renovated, since they will have more powerful GPUs.
- Today, I am working on the CellPose demo model training. It is the 'model_training' notebook. It serves to show how to put the training and testing data into a CellPose model and train it and do some basic evaluation of performance. The more advanced evaluations of performance will be in another notebook for an easier to read demonstration, is the organisation idea I am going with right now. Right now, the demonstration of model training notebook can import the data, display it, train a (pre-trained) model with it and evaluate its performance with the average_precision method from the CellPose package. Tomorrow I will make it display the predictions (masks) on the testing data.

### Friday 24/06/22
- I have not received the confirmation email that GlobalProtect was installed in the lab computer. I am still waiting to be able to use Omero.
- I uploaded the cell data collected with Jesus to the GitHub with the "Large File" option/module from GitHub.
- I finished the model_training notebook, making the predictions, displaying them next to the test data and groundtruth and saving them to a file.
- I started the model_evaluation notebook where I added the methods to make the crops from the predicted masks and will add the qualitative evaluations (binary image operations and crop comparisons) as well as the quantitative operations (number of ROIs, pixel to pixel).

##Small TODOs to delete
- Make GitHub cleaner, add a "How to start this project" part with a how-to on creating a conda environment and the priority of running the "testing_jup" notebook file
- Make a clean DataSpell notebook that can train and evaluate a CellPose2 model, no need for a long walkthrough.
