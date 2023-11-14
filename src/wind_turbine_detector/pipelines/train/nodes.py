import numpy as np
import pandas as pd
import os

import cv2
import re
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm


class TurbineDataset(Dataset):
    """
    Custom PyTorch dataset for turbine image data.

    This dataset is designed for object detection tasks where each image contains
    annotations of turbine components. It can be used for both training and inference.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing image and annotation
            information.
        transforms (callable, optional): A function/transform to apply
            to the image data.
        train (bool, optional): Specify if the dataset is for training (True)
            or inference (False).

    Attributes:
        image_ids (numpy.ndarray): Unique image IDs extracted from the DataFrame.
        df (pandas.DataFrame): The input DataFrame containing image and annotation
            information.
        transforms (callable, optional): A function/transform for image data
            augmentation.
        train (bool): Indicates whether the dataset is for training (True) or
            inference (False).

    Methods:
        __len__(): Returns the number of unique images in the dataset.
        __getitem__(index): Retrieves an image and its associated annotations.

    For training:
    - Images are loaded and transformed.
    - Annotations are retrieved and organized into a dictionary.

    For inference:
    - Only images are loaded and returned.

    Returns:
        If 'train' is True:
            Tuple containing:
                - image (torch.Tensor): The preprocessed image.
                - target (dict): A dictionary containing annotations
                    (boxes, labels, etc.).
                - image_id (str): ID of the image.
        If 'train' is False:
            Tuple containing:
                - image (torch.Tensor): The preprocessed image.
                - image_id (str): ID of the image.
    """
    def __init__(self, dataframe, transforms=None, train=True):
        super().__init__()

        self.image_ids = dataframe['image'].unique()
        self.df = dataframe
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms is not None:
            image = self.transforms(image)
        if self.train is False:
            return image, image_id

        records = self.df[self.df['image'] == image_id]
        boxes = records[['minx', 'miny', 'maxx', 'maxy']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        return image, target, image_id


class Averager:
    """
    A utility class for calculating and maintaining the average of a series of values.

    This class is typically used for computing the average loss during training
    iterations.

    Attributes:
        current_total (float): The running total of values to be averaged.
        iterations (float): The number of values added to the running total.

    Methods:
        send(value): Add a new value to the running total and update the number of
            iterations.
        value: Property that returns the average of the added values.
        reset(): Reset the running total and number of iterations to zero.

    Example Usage:
    ```
    avg_loss = Averager()
    avg_loss.send(2.0)
    avg_loss.send(3.0)
    average = avg_loss.value  # Returns 2.5
    avg_loss.reset()  # Resets the total and iterations to zero.
    ```

    Note:
        If no values are added (iterations = 0), the `value` property returns 0 to
        prevent division by zero.

    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        """
        Add a new value to the running total and update the number of iterations.

        Args:
            value (float): The value to be added to the running total.
        """
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        """
        Get the average value of the added values.

        Returns:
            float: The average value, or 0 if no values have been added
                (iterations = 0).
        """
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        """
        Reset the running total and number of iterations to zero.
        """
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    Collates a batch of data elements into a structured format.

    This function is typically used in data loading pipelines, such as when working with
    PyTorch's DataLoader. It takes a batch of individual data elements and arranges them
    into a structured format, often as a tuple or a dictionary, making it suitable for
    further processing.

    Args:
        batch (list): A list of individual data elements to be collated.

    Returns:
        tuple: A tuple containing the collated data elements. The specific structure
        of the returned tuple may vary depending on the data and the application.

    Example Usage:
    ```
    batch = [(image1, label1), (image2, label2), (image3, label3)]
    collated_batch = collate_fn(batch)
    # Example collated_batch: ((image1, image2, image3), (label1, label2, label3))
    ```

    Note:
        The structure of the returned tuple should match the requirements of the
        downstream processing steps, such as model input.

    """
    return tuple(zip(*batch))


def prepare_batches_for_training(folds: dict, selected_data: gdp.GeoDataFrame, number_of_fold: int):
    trans = transforms.Compose([transforms.ToTensor()])
    train_df = selected_data[selected_data['image'].isin(folds[number_of_fold]['train'])]
    test_df = selected_data[selected_data['image'].isin(folds[number_of_fold]['test'])]
    train_dataset = TurbineDataset(train_df, trans,True)
    test_dataset = TurbineDataset(test_df, trans,True)

    indices = torch.randperm(len(train_dataset)).tolist()
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
