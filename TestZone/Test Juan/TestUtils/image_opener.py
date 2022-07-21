from typing import Any

import numpy as np
import os
from PIL import Image
from numpy import ndarray


def matrix_generation(directory: str, value: int) -> tuple[list[Any], list[int]]:
    x_array, y_array = [], []
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if file != 'desktop.ini':
            pic = Image.open(f).convert('L')
            pic = np.array(pic)
            pic = np.ravel(pic)
            x_array.append(pic)
            y_array.append(value)
    return x_array, y_array


def fill_x_y(x_directory: str, y_directory: str) -> tuple[ndarray, list[int]]:
    x_array, y_array = matrix_generation(x_directory, 1)
    x_array, y_array = matrix_generation(y_directory, -1)
    x_array = np.array(x_array)
    return x_array, y_array
