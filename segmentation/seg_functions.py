# All the code for getting the crop coordinates
import math
from itertools import combinations

import cv2
import numpy as np
from cellpose import metrics, models, io
from cellpose.metrics import boundary_scores, aggregated_jaccard_index
from matplotlib import pyplot as plt


# Function to get the number of cells in an image
# Input: image
# Output: number of cells
# The number of cells is defined by the number of different pixel intensities, excluding the background
def count_cells(img):
    return len(np.unique(img.reshape(-1), return_counts=False)) - 1


# Function to get the 4 coordinates of cell
# Input: binary image where cell is singled out
# Output: list with top left coordinates, bottom left coordinates, top right coordinates, bottom right coordinates
def get_corners(binary):
    top_left_col, top_left_row = None, None
    bot_right_col, bot_right_row = None, None
    top_right_col, top_right_row = None, None
    bot_left_col, bot_left_row = None, None

    # Get the first True's row and column (top left)
    for i in range(len(binary)):
        if True in binary[i]:
            top_left_row = i
            for j in range(len(binary[i])):
                if binary[i][j]:
                    top_left_col = j
                    break
        if top_left_row is not None and top_left_col is not None: break

    # (bottom left)
    for i in range(len(binary)):
        if True in binary[i]:
            while i < 1080 and True in binary[i]: i += 1
            bot_left_row = i - 1
            for j in range(len(binary[bot_left_row])):
                if binary[bot_left_row][j]:
                    bot_left_col = j
                    break
        if bot_left_row is not None: break

    # Get the last True's row and column (bottom right)
    for i in range(len(binary)):
        if True in binary[i]:
            bot_right_row = i
            for j in range(len(binary[i])):
                if binary[i][j]: bot_right_col = j

    # Start reading from the columns (top right)
    for i in range(len(binary)):
        if True in binary[i]:
            top_right_row = i
            for j in range(len(binary[i])):
                if binary[i][j]: top_right_col = j
        if top_right_row is not None and top_right_col is not None: break
    return [(top_left_col, top_left_row), (bot_left_col, bot_left_row), (top_right_col, top_right_row),
            (bot_right_col, bot_right_row)]


# Function to get the center of a cell
# Input: 4 corners of a cell
# Output: coordinates of center of cell

def get_center(corners):
    x_coors = [corners[i][0] for i in range(4)]
    y_coors = [corners[i][1] for i in range(4)]
    center_col = int(sum(x_coors) / len(y_coors))
    center_row = int(sum(y_coors) / len(y_coors))

    return center_col, center_row


# Function to get the longest diagonal
# Input: 4 coordinates of cell
# Output: longest line between two points
def get_longest_line(corners):
    point_combinations = sum([list(map(list, combinations(corners, i))) for i in range(len(corners) + 1)], [])
    point_combinations = [combi for combi in point_combinations if len(combi) == 2]

    longest = 0
    test = []
    for combi in point_combinations:
        current_len = math.hypot(combi[0][0] - combi[1][0], combi[0][1] - combi[1][1])
        test.append(current_len)
        if current_len > longest: longest = current_len

    return int(longest)


# Function that takes a mask and returns a crop of every cell inside it
# Input: image of cells (mask)
# Output: list of images that are crops of the cells in the inputted image
def get_cell_crop_coordinates(original_cells_img, margin=None):
    if margin is None:
        margin = 10

    copy_cells_img = np.copy(original_cells_img)

    # get the number of nuclei
    num_cells = count_cells(copy_cells_img)

    crops = []
    crop_coordinates = []

    # loop through the number of nuclei
    for nuclei_num in range(1, num_cells + 1):
        # get singled out cell image
        cell_img = np.copy(copy_cells_img)
        cell_img = cell_img == nuclei_num

        corners = get_corners(cell_img)  # get the four corners of the nuclei

        center_col, center_row = get_center(corners)  # get the center of the nuclei

        crop_len = int(
            get_longest_line(corners) / 2) + margin  # get the longest diag and divide by 2 for square crop half length

        bottom_row = max(0, center_row - crop_len)
        top_row = min(copy_cells_img.shape[0] - 1, center_row + crop_len)
        bottom_col = max(0, center_col - crop_len)
        top_col = min(copy_cells_img.shape[1] - 1, center_col + crop_len)

        crop_coordinates.append([bottom_row, top_row, bottom_col, top_col])  # store the crop coordinates

        crops.append(cell_img[bottom_row:top_row, bottom_col:top_col])  # make the crop and store it in the list

    return crop_coordinates


# Function to get crops on image from list of coordinates
# Input: original image, list of coordinates to crop around(4)
# Output: crops of these coordinates
def get_img_crops(img, crop_coordinates):
    crops = [img[coord[0]:coord[1], coord[2]:coord[3]] for coord in crop_coordinates]
    return crops


# Function that takes an image, its ground truth mask and its predicted mask and output the comparison between the
# cropped cells it made It should also output the number of cells found by each (later on this should be a measure
# metric)
def match_cell_crops(original_img, gt_mask, pred_mask):
    # Get the crops for the gt mask and the pred mask
    gt_crops = get_img_crops(original_img, get_cell_crop_coordinates(gt_mask))  # lets try and speed up this function

    pred_crops = get_img_crops(original_img, get_cell_crop_coordinates(pred_mask))
    # Sort them to see which correspond to which

    ious = metrics.mask_ious(gt_mask, pred_mask)

    # Match them up into pairs
    pairs = [(gt_crops[i], pred_crops[ious[1][i] - 1]) if ious[1][i] - 1 != -1 else (gt_crops[i], [0]) for i in
             range(len(ious[1]))]

    return pairs


def display_pairs(pairs, ious=None, order="normal", filter=None, num=None):
    # we'll add a sorting method later that sorts by most similar to most dissimilar pairs open up pairs_test to just
    # be a sequential list, no tuples for comparison, print the coordinates of the center of the crop relative to the
    # original img to show how far away these two crops are, we might use that distance as a threshold for matching
    # them up lets add this function to the py file lets make a function that calculates the distance between two crops

    # add some sorting to the pairs
    # first kind should be "normal"
    # second kind should be "most different"
    # third kind should be "most similar"
    # fourth kind should be "only matched"
    # fifth kind should be "only non-matched"
    # pairs = np.array(pairs)

    if ious is not None and filter is not None:
        if filter == 'only_matched':
            pairs_kept = []
            for i in range(len(pairs)):
                if ious[0][i] != 0: pairs_kept.append(pairs[i])
            pairs = pairs_kept
        if filter == 'only_non_matched':
            pairs_kept = []
            for i in range(len(pairs)):
                if ious[0][i] == 0: pairs_kept.append(pairs[i])
            pairs = pairs_kept

    if ious is not None and order != "normal":
        pairs = [pair for _, pair in sorted(zip(ious[0], pairs), key=lambda first: first[
            0])]  # sorted from lowest similarity to highest similarity
        if order == 'most_similar': pairs.reverse()

    pairs_seq = []
    for pair in pairs:
        pairs_seq.append(pair[0])
        pairs_seq.append(pair[1])

    # print(pairs_seq[7]==pairs[3][1])
    # This plotting, for 84 images, takes 3 seconds (the rest of this function takes 0s)
    if num is None:
        num = len(pairs_seq)
    else:
        num *= 2
    plt.figure(figsize=(2, num))
    for i in range(0, num, 2):
        plt.subplot(int(num / 2), 2, i + 1)
        plt.imshow(pairs_seq[i])
        plt.axis('off')
        plt.subplot(int(num / 2), 2, i + 2)
        # print(pairs_seq[i+1])
        if np.array(pairs_seq[i + 1]).all() != 0:
            plt.imshow(pairs_seq[i + 1])
        plt.axis('off')
    plt.show()


# Function to use an imported model to make predictions and returns them
def make_predictions(model_dir, test_dir, gpu=True, channels=None):
    if channels is None:
        channels = [0, 0]
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_dir)

    test_data, test_labels = io.load_train_test_data(test_dir, mask_filter='_seg.npy')[:2]  # loads the test data
    predicted_test_masks = model.eval(test_data, channels=channels, diameter=model.diam_labels.copy())[
        0]  # generates the predictions

    return predicted_test_masks


# Function to display the predictions that I made independent
def display_imgs(imgs):
    plt.figure(figsize=(20, 20))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs) + 1, i + 1)
        plt.axis('off')
        plt.imshow(imgs[i])
    plt.show()


# Function to use an imported model to make predictions on images in a test directory and display them
def make_and_display_predictions(model_dir, test_dir, gpu=True, channels=None):
    if channels is None: channels = [0, 0]
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_dir)

    test_data, test_labels = io.load_train_test_data(test_dir, mask_filter='_seg.npy')[:2]  # loads the test data
    predicted_test_masks = model.eval(test_data, channels=channels, diameter=model.diam_labels.copy())[
        0]  # generates the predictions

    # Displaying the predictions
    plt.figure(figsize=(20, 20))
    for i in range(len(predicted_test_masks)):
        plt.subplot(1, len(predicted_test_masks) + 1, i + 1)
        plt.axis('off')
        plt.imshow(predicted_test_masks[i])
    plt.show()


# Function to get the average precision from ground truth to predicted masks
# This is just a simplification from the CellPose metrics function, it uses it
def get_average_precision(gt_masks, pred_masks, threshold=None):
    if threshold is None:
        threshold = [0.1]
    return metrics.average_precision(gt_masks, pred_masks, threshold=threshold)[0][:, 0].mean()


# Function that takes the ground truth and predicted masks to display the average precision of the model at different
# thresholds
def display_average_precision(gt_masks, pred_masks):
    # list of thresholds we want
    thresholds = [i / 10 for i in range(1, 11, 1)]

    # loop getting those thresholds storing them in a list
    average_precisions = metrics.average_precision(gt_masks, pred_masks, threshold=thresholds)[0]
    results = [average_precisions[:, i].mean() for i in range(len(thresholds))]

    # put the results in a bar plot and display it
    plt.title('Average precision of model at different thresholds')
    plt.bar(np.array(thresholds), np.array(results), width=0.05)
    plt.xticks(np.array(thresholds))
    plt.xlabel('Threshold')
    plt.ylabel('Average precision')
    plt.ylim([0, 1])
    plt.show()


# Function to get the average number of true positives, false positives, and false negatives at different thresholds
def display_average_confusion(gt_masks, pred_masks):
    # list of thresholds we want
    thresholds = np.array([i / 10 for i in range(1, 11, 1)])

    # loop getting those thresholds storing them in a list
    average_tp = metrics.average_precision(gt_masks, pred_masks, threshold=thresholds)[1]
    average_fp = metrics.average_precision(gt_masks, pred_masks, threshold=thresholds)[2]
    average_fn = metrics.average_precision(gt_masks, pred_masks, threshold=thresholds)[3]

    # get the averages
    results_tp = [average_tp[:, i].mean() for i in range(len(thresholds))]
    results_fp = [average_fp[:, i].mean() for i in range(len(thresholds))]
    results_fn = [average_fn[:, i].mean() for i in range(len(thresholds))]

    # put the results in a bar plot and display it
    plt.title('Average true positives, false positives,\nfalse negatives at different thresholds')
    plt.plot(thresholds, np.array(results_tp), 'o', label='True positives')
    plt.plot(thresholds, np.array(results_fp), 'o', label='False positives')
    plt.plot(thresholds, np.array(results_fn), 'o', label='False negatives')
    plt.xticks(np.array(thresholds))
    plt.xlabel('Threshold')
    plt.ylabel('Average metric')
    plt.grid(axis='x')
    plt.ylim([0, max([max(results_tp), max(results_fp), max(results_fn)]) + 5])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
    plt.show()


# Function that displays in bar plot the average precision of multiple models at different thresholds
# Input: a list of models made of tuples with the labels and predictions
def display_average_precision_comparison(model_data):
    thresholds = np.array([i / 10 for i in range(1, 11, 1)])

    # loop getting those thresholds storing them in a list
    averages_per_model = []
    for model in model_data:
        averages_per_model.append(metrics.average_precision(model[0], model[1], threshold=thresholds)[0])

    # get the averages
    results_per_model = []
    for model_average in averages_per_model:
        results_per_model.append([model_average[:, i].mean() for i in range(len(thresholds))])

    # get the spacings for the bars
    space = [-0.04, 0.04]
    total_width = space[1] - space[0]
    width = (space[1] - space[0]) / len(model_data)
    right = width
    bar_centers = []
    for i in range(len(model_data)):
        bar_centers.append(width / 2 + (right - width))
        right += width
    bar_centers = np.array(bar_centers[::-1]) + space[0]

    # put the results in a bar plot and display it
    plt.title('Average precision at different thresholds to compare models')
    for i in range(len(results_per_model)): plt.bar(thresholds - bar_centers[i], np.array(results_per_model[i]),
                                                    width=width, label=str('Model ' + str((i + 1))))
    plt.xticks(np.array(thresholds))
    plt.xlabel('Threshold')
    plt.ylabel('Average precision')
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


# Function to get the mean boundary score between the ground truth and predictions at different scales
# Returns the precision, recall and fscore
def get_average_boundary_scores(gt_masks, pred_masks, scales=None):
    if scales is None:
        scales = [0.1, 0.5]
    results = []  # the results will be stored per scale
    for i in range(len(scales)):
        scores = boundary_scores(gt_masks, pred_masks, [scales[i]])
        result_scale = [scores[0][0].mean(), scores[1][0].mean(), scores[2][0].mean()]
        results.append(result_scale)
    return results


# Function to display the average boundary score in a scatter plot at different scales
def display_average_boundary_scores(gt_masks, pred_masks, scales=None):
    if scales is None:
        scales = [0.1, 0.3, 0.5, 0.7, 0.9]
    scores = np.array(get_average_boundary_scores(gt_masks, pred_masks, scales))
    precision = scores[:, 0]
    recall = scores[:, 1]
    f1 = scores[:, 2]

    plt.title('Boundary scores at different scales')
    plt.plot(scales, np.array(precision), 'o', label='Precision')
    plt.plot(scales, np.array(recall), 'o', label='Recall')
    plt.plot(scales, np.array(f1), 'o', label='F-score')
    plt.xticks(np.array(scales))
    plt.xlabel('Scale')
    plt.ylabel('Average metric')
    plt.grid(axis='x')
    plt.ylim([0, 1.0])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
    plt.show()


def display_average_boundary_score_comparison(model_data, boundary_score=None, scales=None):
    if boundary_score is None:
        boundary_score = 'precision'

    if boundary_score == 'precision':
        ind = 0
    elif boundary_score == 'recall':
        ind = 1
    elif boundary_score == 'f-score':
        ind = 2

    if scales is None:
        scales = [0.1, 0.5, 0.9]

    results_per_model = []
    for gt_masks, pred_masks in model_data:
        scores = np.array(get_average_boundary_scores(gt_masks, pred_masks, scales))
        results_per_model = scores[:, ind]

    # get the spacings for the bars
    space = [-0.04, 0.04]
    total_width = space[1] - space[0]
    width = (space[1] - space[0]) / len(model_data)
    right = width
    bar_centers = []
    for i in range(len(model_data)):
        bar_centers.append(width / 2 + (right - width))
        right += width
    bar_centers = np.array(bar_centers[::-1]) + space[0]

    # put the results in a bar plot and display it
    plt.title('Average ' + boundary_score + ' at different scales to compare models')
    for i in range(len(results_per_model)):
        plt.bar(scales - bar_centers[i], np.array(results_per_model[i]),
                width=width, label=str('Model ' + str((i + 1))))
    plt.xticks(np.array(scales))
    plt.xlabel('Scale')
    plt.ylabel(boundary_score)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


# Function to get the average aggregated jaccard index between the ground truth masks and the predicted ones
def get_average_aji(gt_masks, pred_masks):
    return aggregated_jaccard_index(gt_masks, pred_masks).mean()


# Function to display in a bar plot the average aggregated jaccard index of different models
def display_average_aji_comparison(model_data):
    results_per_model = []
    for gt_masks, pred_masks in model_data:
        results_per_model.append(aggregated_jaccard_index(gt_masks, pred_masks).mean())

    plt.bar([i for i in range(len(model_data))], results_per_model, color=['royalblue', 'bisque', 'olive'])
    plt.ylim([0, 1])
    plt.xticks([i for i in range(len(model_data))], ['Model ' + str(i + 1) for i in range(len(model_data))])
    plt.title('Average Aggregated Jaccard Index')
    plt.ylabel('Aggregated Jaccard Index')
    plt.show()


# Function that returns an image rotated to -90, 90 and 180 degree rotations
def rotate_img(img):
    rotated = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
               cv2.rotate(img, cv2.ROTATE_180)]
    return rotated


# Function to augment the training data images and labels with -90, 90 and 180 degree rotations
def augmentation_rotate(train_data, train_labels):
    train_data_augmented = np.copy(train_data)
    train_labels_augmented = np.copy(train_labels)

    for i in range(len(train_data)):
        # Data (cell images) augmenting
        rotated_ori_data = rotate_img(train_data[i])
        for img in rotated_ori_data:
            train_data_augmented = np.append(train_data_augmented, img)

        horizontal_flip_data = cv2.flip(train_data[i], 0)
        train_data_augmented = np.append(train_data_augmented, horizontal_flip_data)
        rotated_horizontal_flip_data = rotate_img(horizontal_flip_data)
        for img in rotated_horizontal_flip_data:
            train_data_augmented = np.append(train_data_augmented, img)

        # Labels augmenting
        rotated_ori_label = rotate_img(train_labels[i])
        for img in rotated_ori_label:
            train_labels_augmented = np.append(train_labels_augmented, img)

        horizontal_flip_labels = cv2.flip(train_labels[i], 0)
        train_labels_augmented = np.append(train_labels_augmented, horizontal_flip_labels)
        rotated_horizontal_flip_labels = rotate_img(horizontal_flip_labels)
        for img in rotated_horizontal_flip_labels:
            train_labels_augmented = np.append(train_labels_augmented, img)

    train_data_augmented = train_data_augmented.reshape((int(
        train_data_augmented.shape[0] / (train_data[0].shape[0] * train_data[0].shape[1])), train_data[0].shape[0],
                                                         train_data[0].shape[1]))
    train_labels_augmented = train_labels_augmented.reshape((int(
        train_labels_augmented.shape[0] / (train_labels[0].shape[0] * train_labels[0].shape[1])),
                                                             train_labels[0].shape[0], train_labels[0].shape[1]))
    return train_data_augmented, train_labels_augmented


# Function to change the brightness of an image
def change_brightness(img, value):
    # First normalise the image
    img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255

    if value > 0:  # make img brighter
        img_bright = np.where((255 - img_norm) < value, 255, img_norm + value)
    else:  # make img less bright
        img_bright = np.where((0 - img_norm) < value, 0, img_norm + value)
    return img_bright


# Function to augment the training data images with different brightness
# The ground truth labels do not change
def augmentation_brightness(train_data):
    values = [-100, -50, 50, 100]  # the values by which the brightness will change
    train_data_augmented = np.copy(train_data)
    britghtened_imgs = []
    for i in range(len(train_data_augmented)):
        for value in values:
            britghtened_imgs.append(change_brightness(train_data_augmented[i], value))

    for img in britghtened_imgs:
        train_data_augmented = np.append(train_data_augmented, img)

    train_data_augmented = train_data_augmented.reshape((int(
        train_data_augmented.shape[0] / (train_data[0].shape[0] * train_data[0].shape[1])), train_data[0].shape[0],
                                                         train_data[0].shape[1]))
    return train_data_augmented

#%%
