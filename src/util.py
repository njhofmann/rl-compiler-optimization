import random
import numpy
import torch
import os


def set_random_seed(seed: int) -> None:
    """Sets random seed for all available libraries to the given seeds"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
