import dataclasses
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset

CORRUPTIONS = None

def load_cifar10c(
        n_examples: int,
        severity: int = 5,
        data_dir: str = "/gallery_tate/wonjae.roh",
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS
) -> Tuple[torch.Tensor, torch.Tensor]:

    dataset_dir = os.path.join(data_dir, "cifar10c")

    return _load_corruptions_dataset(n_examples, severity, dataset_dir,
                                     shuffle, corruptions, None, None)

def _load_corruptions_dataset(
        n_examples: int, 
        severity: int, 
        data_dir: str, 
        shuffle: bool,
        corruptions: Sequence[str], 
        gdrive_ids: None,
        labels_gdrive_id: str
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download labels
    labels_path = data_dir + '/labels.npy'
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_dir + '/' + corruption + '.npy'
        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                                                           n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test