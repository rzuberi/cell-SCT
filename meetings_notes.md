# Meeting notes

This journal (started on Wednesday the 13th of July) will serve as storage for the notes I take during and after meetings. I find that I take a lot of notes of certain keywords, ideas and recommendations that are given by my supervisors (Dr Helfrid Hochegger and Dr Ivor Simpson) but they are usually in random notes. At least now, they will be in one place, here.

### Wednesday 13th of July, 10:30-11:00
Attendees: Dr Ivor Simpson, Rehan Zuberi
- Talked about where I am right now, what I did, what I have to do until friday, and setting up some time to chart next week about where the project will go (3rd year project and publication and sharing wiht other students such as PhD students in Helfrid's lab who are less Computer Sciency)
- More precisely, I showed Ivor the rules I was applying (the EdU) one and he said that the bins and elbow method to get the background was sensible. He strongly recommended some normalisation of the images before doing this data gathering to make sure there isn't one image that's off that is all classified into one cell.
- Ivor recommended one hot encoding, [0,0,1,0,0] [0,0.5,0.5,0,0] to give the cell cycle phases as some cells can be in transition and we don't want to punish unfairly our classifier for giving a 50/50 chance phase when it is chance and it could just give us the confidence
- I told Ivor that for now I'm applying the rules that Helfrid told me until Friday.
- Ivor wants to setup some time next week to take a step back on the project and make bullet points to know where we go from here. Firstly, how will this project go into being my final year project. Secondly, how can this project and its result be used for publication (useful for him, useful for me). And thirdly, how I can share these resources with other students, especially the ones in Helfrid's lab who are less computer sciency. If they could have an easy to work pipeline in which to put their data and get quantitative results from it, it would be very useful from it.

### Thursday 14th of July, 17:05-17:20
Attendees: Dr Helfrid Hochegger, Rehan Zuberi
- Meeting to show him the progress I've made on the microscopy pictures to make training data for a model
- For the crop displays, he recommends grayscale and enhancing contrast.
- To get all of the cells, he recommends doing a comparison with a predicted mask from the DAPI images. This reminds me to make a model and groundtruth data that works well on segmenting on the kind of data we have now with CellPose2.
- Maybe a size filter for too big or too small objects in the segmentation would be useful.
- Skimage should provide some good tools to help with border images.
- I need to inspect better the EdU and Cyclin A2 values to get the threshold.
- Helfrid wants to see a model soon, hopefully we can get a working CNN by Monday so we can show him a good working pipeline that just has to be improved.
- He asked how the crops were made and this reminds me to make a flowchart/diagram that shows with the diagonal calculations and all how they are made.

### Monday 18th of July, 10:00-10:20
- Have a set of convolutions fo reach channel, the cyclin, and merge them for a classifier or use them idnividually
- Not a classifier for each channel but an encoder. Use 2-3 convolutions and merge them
- More preprocesssing than just normalising is not needed since the CNN will be used
- The images should be interchangeable so the normalisation per dataset might be enough. Make sure its noted that this might need to be digged into more later on.
- Random 90 degree rotations are easy to do with PyTorch according to Ivor.
- PyTorch lightning is the equivalent to Keras. PyTorch for academic research is faster than TensorFlow. TensorFlow in deployment is more used because its good at it but PyTorch is now working for deployment and its faster and flexible and can work on the M1 chips.
- Cuda is the link (software librayr) to link the GPU hardware. Hopefully people will use Docker and more. Installing Docker for Helfrid's lab is worth it (not important for my project, its too much work, but for someone who has more time its worth it). There a Docker open-source equivalent called Singularity
- Helfrid is gone from week 8 wednesday. I talked about giving in the poster in beginning of september so Helfrid wants to be kept in the loop of how the poster goes in august.
- We can move this to the 3rd year project. Time and space tracking of cells as Helfrid suggests. Ivor will talk with Ian Mackie this afternoon. Helfrid will talk with Jesus to get some data with Jesus for tracking. Helfrid will get in touch when he has the data.
- Will have the talk in August or September with the list of extenstion tasks and propose them to Helfrid and Ivor how technical and application of which should be done according to me t build into a publication and for a 3r dyear  project.