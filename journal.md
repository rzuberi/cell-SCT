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
- Made some ground truth data to train a model on.

### Thursday 09/06/22
- Tried a little more to make GPUs work.
- Started a Google Colab notebook to walk-through training a CellPose2 model with channel 1s to segment channel 0.

### Friday 10/06/22
- Working on Colab notebook, importing the data and labelling it.


## Week 2

### Monday 13/06/22
- Meeting with Ivor to present what I currently have (Colab notebook), what I'm planning to have for the next week (a working segmentation model) and my issues with GPU in the lab computers.
- Meeting with Helfrid to talk about what I've done with the data and the Colab segmentation notebook I'm working on.
- Second meeting with Helfrid (in his office) for him to show me Omero, how a pipeline works, and some images, also asked for me to be added to the Omero VPN.

### Tuesday 14/06/22
- Working on Colab notebook to import a model and use it to make predictions.
- Working on Colab notebook to compare predictions and ground truth, binarised the images first (not great direction, Ivor gave better things to look at such as making a bounding box around the ground truth and putting it on the predicted mask to see if it gets the cell, pixel by pixel is the main expression he used).
- End of day quick catch-up with Ivor for him to give me advice on how to get the accuracy of the model (comparing ground truth and predicted masks).

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
- Retried creative a virtual environment to install the GPU version of PyTorch to be able to use the GPU when training CellPose2 human-in-the-loop models, which would be much faster (failed to do this last week). I've been able to install the right version of PyTorch on a virtual environment (that's a win) but unable to make CellPose2 us it (for now). CellPose2 says "[INFO] TORCH CUDA version not installed/working.".
- Working on Colab notebook that demonstrates qualitative testing (with crops) of cell masks. Its added to this GitHub project.
- Added the messy Colab notebook (walk-through that can implement a model, train one and segment) from which the content of will be separated into smaller notebooks to the GitHub project.


## Week 3

### Monday 20/06/22
- Got GPU working for CellPose2 in virtual environment, much faster, very happy.
- Activated education DataSpell license to use it for notebooks instead of Colab.
- Trying to make DataSpell, which uses Anaconda, install the PyTorch GPU to actually use the GPU through the DataSpell notebook. Currently, not working. It seems to create the notebook in the virtual environment, but when executing conda commands from the notebook (not the terminal), it runs in the "base" environment, the one in which I do not have the permissions to install the PyTorch GPU which is needed to run CellPose2 faster.
- Weekly meeting with only Helfrid as Ivor is away. I caught him up on my progress and talked a little about the experience in his lab with Jesus. Shared the GitHub with him (therefore, making it urgent to clean up).

### Tuesday 21/06/22
- Helfried notified me that I should have access to Omero now. With GlobalProtect activated on his computer, with my credentials, he can connect to Omero. I cannot from the lab machine because I do not have GlobalProtect, which would connect me to the internal VPN. An email was sent and Helfrid put the research computing team (that knows about Omero) in contact with Christopher Sothcott who manages the Chichester lab computers.
- Jupyter refused to work (apparently not in the same file) when launched today from DataSpell. I created a new directory in DataSpell and created a Jupyter notebook in there. It was correctly put in the 'workspace' conda environment, and it was connected with the GPU. Basically, yesterday's DataSpell issues seemed to have been solved by creating a new folder in a different directory. I'm still debating how to make these notebooks to make sure they're useful. I'm thinking of making one to just train a CellPose model and just tell the user to put the training and testing data in the right folder immediatly, and use the same to segment it.
- While the research computing team and ITS team are sorting out the Omero issue, I am working on the skeleton of the DataSpell notebooks that will directly connect to Omero. Right now, I am trying to make the notebook, from a new conda environment, install the correct PyTorch and CellPose packages and connect to the GPU (and test if it's all working). Then I will look into how to connect a Python notebook to Omero to get the data from there.

### Wednesday 22/06/22
- The back-and-forth emails for GlobalProtect and Omero seem to have concluded with Luke Igerson looking into installing GlobalProtect into the Chichester Lab PCs.
- I felt a little lost yesterday, not knowing how to get these notebooks going with the correct environment and packages. The solution is a requirements.txt file in our GitHub project. That file then needs to be with the ```pip install -r requirements.txt``` command.
- I spent some time trying to understand how to, from the requirements file, install the GPU activated PyTorch (needing the correct CudaToolkit version installed) in our environment. It now works and is connected to the GPU, but the environment in which the notebook is run must be in the ".conda" folder not the "Anaconda3" one.
- I made the "testing_jup.ipynb" file (for which I'll change the name later) that walks through getting in the correct virtual environment, installing the packages from the requirements.txt file, checking that PyTorch is correctly installed, and checking that PyTorch and CellPose both have access to the GPU.

### Thursday 23/06/22
- Luke Ingerson will install GlobalProtect in two computers in the Richmond 4B9 lab. GlobalProtect will be installed remotely. It will be in the Richmond lab as the Chichester labs have ongoing renovations, to not do the work twice. The computers in the Richmond labs have Nvidia Quadron P1000 4GB GPUs. This should be fine for model training, but if it isn't I will ask Luke Ingerson if he can install GlobalProtect in one of the computers in the Chichester labs after they have renovated, since they will have more powerful GPUs.
- Today, I am working on the CellPose demo model training. It is the 'model_training' notebook. It serves to show how to put the training and testing data into a CellPose model and train it and do some basic evaluation of performance. The more advanced evaluations of performance will be in another notebook for an easier to read demonstration, is the organisation idea I am going with right now. Right now, the demonstration of model training notebook can import the data, display it, train a (pre-trained) model with it and evaluate its performance with the average_precision method from the CellPose package. Tomorrow I will make it display the predictions (masks) on the testing data.

### Friday 24/06/22
- I have not received the confirmation email that GlobalProtect was installed in the lab computer. I am still waiting to be able to use Omero.
- I uploaded the cell data collected with Jesus to the GitHub with the "Large File" option/module from GitHub.
- I finished the model_training notebook, making the predictions, displaying them next to the test data and ground truth and saving them to a file.
- I started the model_evaluation notebook where I added the methods to make the crops from the predicted masks and will add the qualitative evaluations (binary image operations and crop comparisons) as well as the quantitative operations (number of ROIs, pixel to pixel).


## Week 4

### Monday 27/06/22
- Had a 10am meeting with Ivor where I explained to him where I was at with the qualitative comparison of models and my questions with how to index. He recommended I explore how CellPose2's loss function works, which I tried reading about today and set a meeting at 11:30, so we can read its source code together.
- Had a 3pm meeting with Ivor and Helfrid to keep them updated on my progress. Helfrid reported some cell images done with Jesus were out of focus, so he will redo them.
- I am still waiting on the confirmation email for the installation of GlobalProtect. After the 3pm meeting with Helfrid and Ivor, they recommended I send them another email, to which Luke Igerson promptly responded that he will chase up the people responsible for the installation.
- I have reorganised the GitHub project to put the big Colab notebooks in an "archive" folder and put the clean ones in the "segmentation" folder. I also removed other small tests and unnecessary files. I also updated the readme to reflect the current state of this project. I also renamed the segmentation notebooks to reflect that they are part of the segmentation part of this project, and I put them in the "segmentation" folder.
- I have explored some of CellPose2's metric functions which seem to offer most of what we were looking for to evaluate the model. There is even a function to find which masks from the predictions and ground truth match most together, creating the pairs. I played around with them a little but need to do some more explorations. The end goal would be to understand fully how they work to implement them into the seg_model_evaluation notebook.

### Tuesday 28/06/22
- Meeting with Ivor to talk about CellPose2's loss function and the metric functions they offer. For the loss function, he recommended I explore the labels and what they actually contain (e.g flow masks) after we read through some source code together. He also proposed that I just use CellPose's metric functions to evaluate the model quantitatively as to save time.
- Omero is now installed in the Richmond lab 4B9 computers (2 of them). I have been able to connect to it with my credentials. I haven't explored it much but there are no images in it as far as I can tell. Tomorrow I will make a notebook to setup Omero to the environment and use it in Python. Maybe in the "environment_setup" notebook as an optional setup.
- I worked on the model evaluation notebook, matching up the cropped ground truth cells to the cropped predicted cells using the cropped functions (that I moved into a separate .py file that is imported into the notebook then to make it cleaner) and the 'mask ious' function from CellPose that finds the best matches in masks. I changed the get_crops_img function to only take the image and find the coordinates to get the crops from there but that was a mistake (that I will fix tomorrow) since I need to get the coordinates from the mask and make the crops on the real microscopy image of the cells.

### Wednesday 29/06/22
- Today I spent all of my time working on the qualitative comparison methods. These methods can pair up the crops of cells on the original image taken from the ground truth masks and the predicted masks and display them next to each other. These methods should be eventually moved from the seg_model_evaluation notebook to the crop_cells.py python file (for which I'll change the name to be more general at some point I think). These functions can also sort how similar the pair of crops is (if the pair really does match up) or by how dissimilar they are. It can also filter to only show the pairs that were found (sometimes the prediction doesn't find all the cells) or only the ones that were not found. I also worked on getting the "count cells" function to be faster, and it is much faster thanks to a numpy built-in function. Python loops really are slow.
- I wish, for tomorrow morning, to clean up the qualitative methods (put them in crop_cells.py and show examples of usage in the notebook), make a plan for the quantitative methods and try to connect Omero to the project. I don't think connecting Omero to the notebook is useful right now since I cannot see any data in it, but at least reading about it will help me gain more information (and go more quickly) during the meeting with Helfrid (tomorrow at 2:30) where he is going to show me how Omero works, I do not want to show up with no knowledge at all.

### Thursday 30/06/22
- Cleaned up the qualitative comparison methods, that's done now in the seg_model_evaluation notebook.
- Added some qualitative methods in the seg_model_evaluation notebook adn cleaned them up in functions, where some display the data in nice plots, in the crop_cells python file. Right now the average precision can be displayed with another function that displays to compare models, and there's an average confusion method to get the average true positives etc. I'll add more of these tomorrow.
- The meeting with Helfrid for Omero was moved to Friday (tomorrow) at 2:30.

### Friday 01/07/22
- Finished adding the quantitative methods to the model evaluation notebook and crop cells py file (for which I should change the name). The py file holds the functions' code, the notebook presents them cleanly.
- Meeting with Helfrid did not happen, moved to Monday (no time defined yet).
- Read about connecting Omero to python and made a quick setup notebook called omero_setup.


## Week 5

### Monday 04/07/22
- Changed the "crop_cells.py" file's name to "seg_functions.py" as it is more fitting to its purpose.
- I started a data augmentation notebook. It takes images and their ground truth and makes more data from it. Today I added the rotation and brightness changes augmentations, with the functions being in the seg_functions python file. There will also be a section in there to measure the effects of the augmentations on the evaluation metrics of the model by training two models, one with the augmented data and one without, and evaluating them.
- Deleted some unused branches from the GitHub project (can be restored later, right now they're useless).

### Tuesday 05/07/22
- Weekly catch-up with Helfrid and Ivor, main point is the switch to the classification part of the project. They gave me good pointers about how to make the training data for the classification CNN and Helfrid showed us how to get the data in Omero. Helfrid will also send me the notebook that can iteratively get data from Omero into a numpy array. I will put, as Ivor recommended, all the points of that meeting into a new section in GitHub with objectives and suggestions.
- Created the classification directory and added the agenda that holds TODOs, current project state, suggestions and issues.
- I worked on the Omero setup notebook trying to import cell images from Omero with the functions that Helfrid gave me. The functions work well, but they are slow (0.22 seconds per image, for 160 images). I tried using NumPy arrays to make it faster but the problem is the built-in Omero function. I looked online and there may be some solutions that use different formats and/or compress the images, I haven't been able to find proper documentation to do that in Python. I might leave this task for later as it took me a lot of my time today and I was unable to speed it up at all. The omero_setup notebook is a mess (of NumPy testing) that I will have to clean up. 

### Wednesday 06/07/22
- Made some very good progress today, quite proud.
- I continued reading about Omero and how to speed up the imports and ended up finding the 'ezomero' python package which makes importing simpler, but not faster. Still, I started using that for any Omero functionalities in the Omero_setup notebook.
- I worked in the Omero_setup notebook to start the classification part to make ground truth data of classified cells. I made functions that take an image from Omero, makes a prediction of the cells in there, crops around them in the DAPI and Cy5 (Edu) channels, finds the average intensity of pixels in both channels in each crops to have 2 values per cells and then k-means clusters in 3 groups to find the G1, S and G2/M phases. There's also some display functions for ease of viewing. When this pipeline is improved with a segmentation model trained on more data and probably some corrections on the k-means clustering to find the right classes (probably because of how the values are calculated with just pixel intensity, there may be a different way to calculate the DAPI and Cy5 values, I need to ask Ivor) this will serve to make ground truth data for the cell cycle labelling CNN. That CNN should classify with only one marker which I forgot the name of right now.

### Thursday 07/07/22
- Slower progress today as I had to work from home, haven't yet tried Omero from home.
- Helfrid told me that Jesus will finish labelling data today, so it should be available tomorrow (Friday) morning. Hopefully he labelled the cell cycle phase of cells, which would provide the base data for my classifier to make groundtruth data for the CNN that will only use one type of labelling. Otherwise it might be just new images of cells. Anyway, I told Helfrid, over Slack, about what I did yesterday and how far I got into making ground truth data, that is having a working pipeline for it right now.
- I read the paper "CNN for Clasifying Chromatin Morphology in Live-Cell Imaging" that was sent by Helfrid on slack a few weeks ago. I learned that some of the methods I used to train the segmentation models (uploading images in google colab directories) are also used by this paper, which is reassuring. This paper uses CNNs for this kind of classification, which is reasurring for my own thinking of using CNN for cell cycle labelling. They also found an accuracy of minimum 95% which is promising for my project. Otherwise there were some techniques/methods that I could apply to my own project later on, such as UMAP for feature filtering.

### Friday 08/07/22
- Found conflicts in the environment setup notebook with MacOS, the ZSH terminal doesn't have the same terminal commands as Windows. Need to fix this.
- I could not launch Omero from my Mac computer because I could not log into GlobalProtect. I think this is due to me not having Sussex staff VPN credentials.
- Helfrid told me that Jesus added more cell data (I'm guessing microscopy pictures) that are 10X to Omero, and that he will make/add 20X data Monday.


## Week 6

### Monday 11/07/22
- Meeting with Helfrid where he gave some recommendations for the classification and said more data of cell images (at 20x) would arrive. Mainly, he said to make the segmentations on the PCNA (Alexa555) channel as that is what will be used in the end (these segmentations are the ones for the mask prediction for the classification and the ones to show the results of the classification). He also gave me some code to normalise the pixel intensities of the cells and some rules to apply (certain value thresholds) that determine the cell cycle phase of cells.
- I was having problems with GlobalProtect (I could not log in because I did not have a VPN password from MyView since I do not have employee credentials). Therefore, I used data (cell image channels) I had already downloaded and put in the classification directory.
- I made a new notebook called ccc_data_exploration where I started and will be putting the code to show how to get the different channels and classify with k-means clustering. I will eventually clean up the omero_setup notebook to not have any of the classification code as it should be kept for the ccc_data_exploration notebook now.
- I learned how to generate and use GitHub personal access tokens to push my work from my Mac.

### Tuesday 12/07/22
- Worked on the ccc_data_exploration notebook. Put the cell data in a Pandas dataframe.
- Worked on applying the rules to the cell dataframe to classify the cells (for the ground truth data). I applied the one for the S-phase and added the column with True or False for fitting the criteria to be in S-phase. This organisation is better than having one 'Phase' column which would prevent observing if one cell fits multiple criteria of phases (could fit the S and the G1 phase). Adding that column works now.
- When applying the rule the color of the background to differentiate the cells that are in the S phase and not is harder. I plotted the cell pixel intensities on the EdU channel with varying bucket numbers and tried getting the elbow which should be the separation between the background and the appearing cells in that channel. The function to get the background is okay for now when not averaging with many buckets (around ~400 pixel intensity). Averaging over many buckets gives lower values (around ~300 pixel intensity).
- Next up I should clean up what I have to apply that S phase rule and make specific functions to apply each rule to classify in the cell phases. Later on I also need to figure out how to apply k-means to this, maybe on a 4 axis graph since we have 4 channels?

### Wednesday 13/07/22
- Meeting with Ivor where I told him about the rules I was applying, he said the background finding method I was using was sensible. Ivor gave me some recommendations for normalisation and one hot encoding. Ivor wants to talk next week about where the project will go, third year project, potential publication and sharing the pipelines and code I made with other cell biology students.
- I made a new markdown document called meeting_notes where I will be putting in there the dates and times where I had meetings with the notes I took in them.
- I continued exploring the EdU classification and did some cleaning up in the ccc_data_exploration notebook. I also made a full plan for the pipeline to use the data in Omero to apply the rules on each. Hopefully I'll be able to finish the pipeline well enough tomorrow to show Helfrid in the afternoon.

### Thursday 14/07/22
- I was able to apply all the classification rules on the cell data. The pipeline to do so is not clean and could be faster. It should also be done on more data.
- Had a meeting with Helfrid where I showed him the data I have with the rules he gave me to apply. He recommended to better explore the Cyclin A2 and EdU markers in how to separate them from the active ones in those channels to the background/negative ones. He also said the clear-cut between cells where none of them (in the data I had, which should be more on more images) were applying to 2 rules, each only 1 rule so only 1 phase classified, is normal.
- I started making a CNN just to get some model running but had some troubles with the image shapes.
- The main priority right now is to make sure that the Cyclin A2 threshold and EdU threshold are better, add more data to get more training data, to get a working CNN pipeline (even if the accuracy is not good yet), to clean up the pipeline and to start making the presentation for Wednesday.

### Friday 15/07/22
- I looked at the EdU and Cyclin A2 thresholds and plotted them. I found that they were finding correct threshold values to separate the data as Helfrid was showing me in his plots. I sent these plots to Helfrid and Ivor on Slack, so they could tell me (especially Helfrid) if they are correct.
- I cleaned up the pipeline by putting the functions in their own py file (ccc_function.py). I made a function to build the dataframe will all the data for multiple images to make a large dataframe. I added the first 50 images only as doing all the steps to get the data for all this took almost 10 minutes. At some point I will go and refactor it. I saved that data as a CSV file. I also got some stats on it for the cell cycle phase proportions and the G2/M grew to ~11% so that looks good, I also sent that to Helfrid and Ivor. I want to put that data into the Venn diagram and if there is no overlap into a pie chart to also show the ~4% of cells that are not labelled at all. I also need to display some of these cells, although I feel like that is less important.
- I really need to update the GitHub README to reflect the state of the project as well as the classification agenda with what I have now. I also really need to make flowcharts. I feel like all of this will come Tuesday when I make the presentation for Wednesday afternoon.
- I restarted making a CNN from a tutorial that Mae sent me and I am starting by making proper directories of training and testing data as the tutorial shows, it is not done yet. I'm hoping to have a model Monday to show Helfrid and Ivor some classification-model data. This will be continued Monday.


## Week 7

### Monday 18/07/22
- Weekly meeting with Ivor and Helfrid. Talked about the current state of the project, how I was able to get the training data and am moving on to classifying with a CNN. Helfrid is excited for how many cells I will have in total and how well the CNN will perform. Helfrid wants me to keep him updated on the performance this week. I need to meet with Ivor before next Monday for him to check how my CNN is (code and performance wise). I will need to talk with them during September about the extension tasks of this project I want to take on for my 3rd year project. I will also need to keep Helfrid updated on the poster in August.
- I worked on getting the CNN to work, it does, but with an accuracy of ~53%. This is fine because there is only 3% of our data used to train it and especially the layers chose are very simple and almost random. With lots of tweaking needed, this model will marginally improve.
- I cleaned up the ccc_data_exploration notebook. It holds too many different parts of the project and rough code that isn't put in py files. It needs more cleaning up once the pipeline is solid and I've sorted which functions I'm keeping. It will need problaly 2 to 3 different py files as I do not want the CNN functions and training data creation functions to be together.
- I also launched the dataframe creating of the entire plate 822, that's 1500 images, it should take 4 hours and will be done tonight but I will see if it worked tomorrow morning. It will save it as a CSV. Then, tomorrow, I will do the same on the computer to my right for it to create it for another plate that Helfrid had given us. That way we will have the two CSVs and can even create a mega one or train two different models on them with the same architecture to observe the differences.
- Tomorrow I will do the CNN in the morning and the presentation in the afternoon. This might also give me an excuse to clean up the GitHub.

### Tuesday 19/07/22
- I did not work on the CNN but I created the data into dataframes from all images in plates 821 and 822 and put them in OneDrive because GitHub is annoying with big files. I made a data_creation notebook that creates this dataframe from the plate but it takes a lot of time (4 hours for 1500 images).
- I worked on the presentation but need to do a lot more tomorrow morning. I want to have supporting images to show, for example, how CellPose2 works to create a segmentation model, how I get the crops with the diagonals, how the classification rules work etc. It is on Google Slides.
- DataSpell crashed a few times in a row because of a lack of memory in the PC, I had to restart it but no data/code was lost. It was probably due to being on all night and that I made it generate so many images in one go. Works fine now.

## TODOs
- Classification parts of omero_setup notebook belong in their own notebook in the classification directory
- Add in omero_setup a 'get_ground_truth' function that gets the gray_crops and the labels found with k-means. Maybe the gray_crops will be different because we'll be using a different marker to train the CNN on.
- Clean up omero_setup
- Add some images (cell images, crops and masks) and flowcharts to the GitHub readme
- Clean up the model_training notebook
- Meeting at 2:30 with Helfrid for ground truth data and Omero data access tutorial (date unsure)
- Update GitHub readme with what works now (such as the qualitative methods)
- Add a "sources' section to the GitHub readme with CellPose2
- How many layers and what size for a CNN, something to research to prepare for classification part of the project
- Could add in the qualitative segmentation a way to compare how models cropped cells differently, using CellPose's cell matching function we could match them all up and then present them each on a row.
- I'll most probably have to make data loaders
- The data augmentation notebook (for the segmentation) is not finished, but can be put on the side for now until I finish setting up the data for the CNN. It will be needed to get a good segmentation model, mostly everything is set up to get a good segmentation model, I just haven't trained a good one yet (because I need to make more segmentation ground truth data).
