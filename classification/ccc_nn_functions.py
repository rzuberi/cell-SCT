import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from tqdm.notebook import trange
from tqdm.notebook import tqdm
from torch import optim
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
import re
import ast
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


# Most of the code here comes from: https://colab.research.google.com/github/bentrevett/pytorch-image-classification
# /blob/master/2_lenet.ipynb#scrollTo=EWBrruYJpJOO

class CCCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, channel_names, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.cell_dataframe = pd.read_csv(csv_file)

        # Add the labels
        # self.cell_dataframe = add_cell_label_as_num_row(self.cell_dataframe)
        # indices_to_skip_label_3 = [i for i in range(len(self.cell_dataframe)) if self.cell_dataframe['label'][i] == 3]
        # print(indices_to_skip_label_3)
        # print(len(self.cell_dataframe))
        # self.cell_dataframe = self.cell_dataframe.drop(indices_to_skip_label_3).reset_index(drop=True)
        # print(len(self.cell_dataframe))

        # Skip the strange images
        # indices_to_skip_img_wrong_shape = [i for i in range(len(self.cell_dataframe)) if str2array(df['pcna_crops'][i]).dtype is np.dtype('object')] #skipping rows with shapes such as (7,)
        # self.cell_dataframe = self.cell_dataframe.drop(indices_to_skip_img_wrong_shape).reset_index(drop=True)
        self.channel_names = channel_names
        self.column_name = channel_names[0]

        df = pd.read_csv(csv_file)
        df = add_cell_label_as_num_column(df)
        indices_to_skip_label_3 = [i for i in range(len(df)) if df['label'][i] == 3]
        df = df.drop(indices_to_skip_label_3).reset_index(drop=True)

        indices_to_skip_img_wrong_shape = [i for i in range(len(df)) if
                                           str2array(df[self.column_name][i]).dtype is np.dtype(
                                               'object')]  # skipping rows with shapes such as (7,)
        df = df.drop(indices_to_skip_img_wrong_shape).reset_index(drop=True)
        self.cell_dataframe = df
        self.transform = transform

    def __len__(self):
        return len(self.cell_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = []
        for i in range(len(self.channel_names)):
            images.append(str2array(self.cell_dataframe[self.channel_names[i]][idx]))

        label = self.cell_dataframe['label'][idx]

        images_transformed = []
        if self.transform:
            convert_tensor = transforms.ToTensor()
            for image in images_transformed:
                image = convert_tensor(image)
                image = self.transform(image)
                images_transformed.append(image)

        images_transformed = torch.Tensor(np.array(images_transformed))

        return images_transformed, torch.tensor(label).long()


def add_cell_label_as_num_column(df):
    conditions = [
        (df['G1_Phase'] == True),
        (df['S_Phase'] == True),
        (df['G2_M_Phase'] == True),
        (df['S_Phase'] == False) & (df['G1_Phase'] == False) & (df['G2_M_Phase'] == False)
    ]
    values = [0, 1, 2, 3]
    df['label'] = np.select(conditions, values)  # applying the conditions to our rows
    return df


def str2array(s):
    # source:https://stackoverflow.com/questions/35612235/how-to-read-numpy-2d-array-from-string/44323021#44323021
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


class LeNet(nn.Module):
    def __init__(self, output_dim, channel_names):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=len(channel_names),
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        # x = [batch size, 1, 28, 28]
        x = self.conv1(x)
        # x = [batch size, 6, 24, 24]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 6, 12, 12]
        x = F.relu(x)
        x = self.conv2(x)
        # x = [batch size, 16, 8, 8]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 16, 4, 4]
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        # x = [batch size, 16*4*4 = 256]
        h = x
        x = self.fc_1(x)
        # x = [batch size, 120]
        x = F.relu(x)
        x = self.fc_2(x)
        # x = batch size, 84]
        x = F.relu(x)
        x = self.fc_3(x)
        # x = [batch size, output dim]
        return x, h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model = model.float()
    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x.float())

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_loss = 0
    epoch_acc = 0

    model = model.float()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x.float())

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_iterators(csv_path, channel_names):
    data = CCCDataset(csv_file=csv_path,
                      channel_names=channel_names,
                      transform=transforms.Compose([CenterCrop([32, 32])]))

    batch_size = 64

    # Splitting the training and testing data
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    n_train_examples = int(len(train_dataset) * 0.9)
    n_valid_examples = len(train_dataset) - n_train_examples

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [n_train_examples, n_valid_examples])

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=batch_size)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size)

    print(
        f'Training examples: {len(train_dataset)} | Validation example: {len(valid_dataset)} | Testing examples: {len(test_dataset)}')
    return train_iterator, valid_iterator, test_iterator


def train_ccc_model(epochs, path_to_save_model, iterators, channel_names):
    train_iterator, valid_iterator, test_iterator = iterators

    # Defining the model

    OUTPUT_DIM = 3

    model = LeNet(OUTPUT_DIM, channel_names)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Training the model
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    print('Torch Cuda is available:', torch.cuda.is_available())

    EPOCHS = epochs

    best_valid_loss = float('inf')

    for epoch in trange(EPOCHS, desc="Epochs"):

        start_time = time.monotonic()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), path_to_save_model)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


# Function to load a model
def load_model(path_to_model):
    OUTPUT_DIM = 3
    model = LeNet(OUTPUT_DIM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_model))
    return model


def get_predictions(model, iterator):
    model = model.float()
    model.eval()

    images = []
    labels = []
    probs = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred, _ = model(x.float())

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Cell cycle classification confusion matrix')
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=['G1', 'S', 'G2/M'])
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.savefig('confusion matrix')

# %%

# %%
