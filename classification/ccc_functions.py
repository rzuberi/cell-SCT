# This file holds the functions to make the necessary classifcation of cells
# Right now, this only has the functions to make the training data

import numpy as np
import numpy.matlib
# Need a function to import all images from a well
import pandas as pd
from cellpose import models
from matplotlib import pyplot as plt
from statistics import mean

from segmentation.seg_functions import get_cell_crop_coordinates, get_img_crops


import sys
#source:https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count),
              end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def normalise_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def get_elbow(curve):
    # source:https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


# This function splits an array into the inputted number and returns the average per split
def get_avg_split_arr(arr, num_splits):
    array_split = np.array_split(arr, num_splits)
    averages = [mean(array) for array in array_split]
    return averages


def get_EdU_threshold(df, column_name):
    edu_nums = np.sort(
        np.array(df.loc[:, [column_name]]).flatten())  # extract EdU values and put them in one array shape num_of_cells
    # Get the list of EdU values
    # Make a split from 3 to the total number of cells
    # Get the averages at that split
    # Get the elbow in those averages
    # Store that elbow in a list
    # Average out the list of elbow and return that value
    elbows_y = []
    start = round(0.73*len(edu_nums)) #we don't want to average out all of the cells because that would create a much a lower value so we only keep the top quarter
    for i in range(start, len(edu_nums) + 1):
        avg_split_arr = get_avg_split_arr(edu_nums, i)
        elbow_x = get_elbow(avg_split_arr)
        elbows_y.append(edu_nums[elbow_x])
    return mean(elbows_y)


def dna_norm(df, column_name=None):
    """
    normalise histogram of DNA label
    :param df: input dataframe
    :return: input dataframe with additional column: "DNA_content"
    Author/Shared by: Dr Helfrid Hochegger
    """
    if column_name == None: column_name = 'dapi_values'
    y, x = np.histogram(df[column_name], 250)
    plt.close()
    max = x[np.where(y == y.max())]
    df['DNA_content'] = df[column_name] / max[0]
    return df


def apply_S_phase_rule(df):
    """
    This function adds a boolean column to indicate if a cell is in the S-phase
    :param df: input dataframe where each row is a cell with a 'EdU_pi' column
    :return: input dataframe with additional column 'S_phase'
    """

    # Step 1: calculate the background color this serves as the threshold for the S-phase
    EdU_threshold = get_EdU_threshold(df, 'edu_values')

    # Step 2: Use that threshold in each row to see if a cell is above or below
    # S_phase_cells = df.loc[df['edu_values'] >= EdU_threshold, 'edu_values']

    df['S_Phase'] = np.where((df['edu_values'] >= EdU_threshold), True, False)

    # df["S_Phase"] = "False"
    # df.loc[df['edu_values'].isin(S_phase_cells), "S_Phase"] = "True"

    return df


def apply_G1_phase_rule(df):
    """
    This function adds a boolean column to indicate if a cell is in the G1-phase
    :param df: input dataframe where each row is a cell with a 'EdU_pi' column
    :return: input dataframe with additional column 'S_phase'
    """

    # Step 1: calculate the background color this serves as the threshold for the S-phase
    cyclina2_thresh = get_EdU_threshold(df, 'cyclina2_values')
    edu_thresh = get_EdU_threshold(df, 'edu_values')

    # Step 2: Use that threshold in each row to see if a cell is above or below

    df['G1_Phase'] = np.where(
        ((df['cyclina2_values'] < cyclina2_thresh) & (df['edu_values'] < edu_thresh) & (df['DNA_content'] < 1.5)), True,
        False)

    return df


def apply_G2_M_phase_rule(df):
    """
    This function adds a boolean column to indicate if a cell is in the G1-phase
    :param df: input dataframe where each row is a cell with a 'EdU_pi' column
    :return: input dataframe with additional column 'S_phase'
    """

    # Step 1: calculate the background color this serves as the threshold for the S-phase
    cyclina2_thresh = get_EdU_threshold(df, 'cyclina2_values')
    edu_thresh = get_EdU_threshold(df, 'edu_values')

    # Step 2: Use that threshold in each row to see if a cell is above or below

    df['G2_M_Phase'] = np.where(
        (((df['DNA_content'] > 1.5) & df['cyclina2_values'] > cyclina2_thresh) & (df['edu_values'] < edu_thresh)), True,
        False)

    return df


def build_one_cell_df(image, model_dir):
    # For now build the dataframe for one image
    dapi_img = np.reshape(image[1], (1080, 1080, 4))[:, :, 0]  # channel 0 is Hoechst 33342, very similar to DAPI
    edu_img = np.reshape(image[1], (1080, 1080, 4))[:, :, 1]  # channel 1 is Cy5 aka EdU
    cyclina2_img = np.reshape(image[1], (1080, 1080, 4))[:, :, 2]  # channel 2 is Alexa488 aka Cyclin A2
    pcna_img = np.reshape(image[1], (1080, 1080, 4))[:, :, 3]  # channel 3 is Alexa555 aka PCNA

    dapi_img = normalise_img(dapi_img)
    edu_img = normalise_img(edu_img)
    cyclina2_img = normalise_img(cyclina2_img)
    pcna_img = normalise_img(pcna_img)

    model = models.CellposeModel(gpu=True, pretrained_model=model_dir)
    test_data = [pcna_img]
    predicted_test_masks = model.eval(test_data, channels=[0, 0], diameter=model.diam_labels.copy())[
        0]  # generates the predictions # we could use the fact that it has a channel input to have the normal image
    # not separated by channel and just make it do the prediction on there, although we need it to make a prediction
    # on the averaged out gray image so that wouldn't really work
    predicted_mask = predicted_test_masks[0]

    crop_coordinates = get_cell_crop_coordinates(predicted_mask, margin=3)
    dapi_crops = get_img_crops(dapi_img, crop_coordinates)
    edu_crops = get_img_crops(edu_img, crop_coordinates)
    cyclina2_crops = get_img_crops(cyclina2_img, crop_coordinates)
    pcna_crops = get_img_crops(pcna_img, crop_coordinates)

    # Step 4: Get the pixel intensity data from each channel and store them in the dataframe
    dapi_values = []
    edu_values = []
    cyclina2_values = []
    pcna_values = []
    for i in range(len(edu_crops)):
        dapi_values.append(np.average(dapi_crops[i]))
        edu_values.append(np.average(edu_crops[i]))
        cyclina2_values.append(np.average(cyclina2_crops[i]))
        pcna_values.append(np.average(pcna_crops[i]))

    dapi_values = np.array(dapi_values)
    edu_values = np.array(edu_values)
    cyclina2_values = np.array(cyclina2_values)
    pcna_values = np.array(pcna_values)

    cell_data = {'dapi_crops': dapi_crops,
                 'edu_crops': edu_crops,
                 'cyclina2_crops': cyclina2_crops,
                 'pcna_crops': pcna_crops,
                 'dapi_values': dapi_values,
                 'edu_values': edu_values,
                 'cyclina2_values': cyclina2_values,
                 'pcna_values': pcna_values}
    df = pd.DataFrame(cell_data)

    df = dna_norm(df)

    df = apply_S_phase_rule(df)

    df = apply_G1_phase_rule(df)

    df = apply_G2_M_phase_rule(df)

    return df


def build_mega_cell_df(images, model_dir):
    pd_list = []
    for i in progressbar(range(len(images))):
        pd_list.append(build_one_cell_df(images[i], model_dir))

    df_concat = pd.concat(pd_list)
    df_concat = df_concat.reset_index() #otherwise the dfs will be separated by their old index, which is still available as another column with this function

    return df_concat


def count_phases(df):
    num_total_cells = df.S_Phase.value_counts()[0] + df.S_Phase.value_counts()[1]
    num_in_s = df.S_Phase.value_counts()[1]
    percentage_in_S = round(df.S_Phase.value_counts()[1] * 100 / num_total_cells, 2)
    num_in_g1 = df.G1_Phase.value_counts()[1]
    percentage_in_g1 = round(df.G1_Phase.value_counts()[1] * 100 / num_total_cells, 2)
    num_in_g2_m = df.G2_M_Phase.value_counts()[1]
    percentage_in_g2_m = round(df.G2_M_Phase.value_counts()[1] * 100 / num_total_cells, 2)

    print('Total cells: ' + str(num_total_cells))
    print('S phase: ' + str(percentage_in_S) + '%')
    print('G1 phase: ' + str(percentage_in_g1) + '%')
    print('G2/M phase: ' + str(percentage_in_g2_m) + '%')




#%%
