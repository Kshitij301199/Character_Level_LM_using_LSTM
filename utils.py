"""
This module provides utility functions for text generation using RNN-based models.

Functions:
    - load_dataset(path: str) -> str:
        Load and return the content of a text file located at the specified path.

    - random_chunk(file_path: str = TRAIN_PATH) -> str:
        Generate a random chunk of text from the loaded dataset.

    - char_tensor(strings: str) -> torch.autograd.Variable:
        Convert a string of characters into a PyTorch tensor.

    - random_training_set(file_path: str = TRAIN_PATH) -> Tuple[torch.autograd.Variable, torch.autograd.Variable]:
        Generate a random training set consisting of input and target tensors.

    - time_since(since: float) -> str:
        Helper function to print the elapsed time since a given timestamp.

Constants:
    - CHUNK_LEN: int
        The length of chunks used for training and generation.
    
    - TRAIN_PATH: str
        The path to the training dataset file.
"""

import unidecode
import string
import random
# import re
import time
import math
import torch

from torch.autograd import Variable

CHUNK_LEN = 200
TRAIN_PATH = './data/dickens_train.txt'


def load_dataset(path):
    """
    Load and return the content of a text file located at the specified path.

    Args:
        path (str): The path to the text file.

    Returns:
        str: The content of the text file.
    """
    all_characters = string.printable
    n_characters = len(all_characters)
    file = unidecode.unidecode(open(path, 'r').read())
    return file


def random_chunk(file_path=TRAIN_PATH):
    """
    Generate a random chunk of text from the loaded dataset.

    Args:
        file_path (str): The path to the training dataset file. Default is TRAIN_PATH.

    Returns:
        str: A random chunk of text.
    """
    file = load_dataset(file_path)
    start_index = random.randint(0, len(file) - CHUNK_LEN - 1)
    end_index = start_index + CHUNK_LEN + 1
    return file[start_index:end_index]


def char_tensor(strings):
    """
    Convert a string of characters into a PyTorch tensor.

    Args:
        strings (str): The input string.

    Returns:
        torch.autograd.Variable: A PyTorch tensor representing the input string.
    """
    all_characters = string.printable
    tensor = torch.zeros(len(strings)).long()
    for c in range(len(strings)):
        tensor[c] = all_characters.index(strings[c])
    return Variable(tensor)


def random_training_set(file_path=TRAIN_PATH):
    """
    Generate a random training set consisting of input and target tensors.

    Args:
        file_path (str): The path to the training dataset file. Default is TRAIN_PATH.

    Returns:
        Tuple[torch.autograd.Variable, torch.autograd.Variable]: Input and target tensors.
    """
    chunk = random_chunk(file_path=file_path)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp.unsqueeze(0), target.unsqueeze(0)


def time_since(since):
    """
    Helper function to print the elapsed time since a given timestamp.

    Args:
        since (float): The starting timestamp.

    Returns:
        str: A formatted string representing the elapsed time.
    """
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
