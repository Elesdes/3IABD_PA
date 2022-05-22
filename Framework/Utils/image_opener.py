import numpy as np
import os
from PIL import Image

def fill_x_and_y(x_dir, y_dir):
    x = []
    y = []
    for filename in os.listdir(x_dir):
        f = os.path.join(x_dir, filename)
        if filename != 'desktop.ini':
            if os.path.isfile(f):
                img = Image.open(f).convert('L')
                img = np.array(img)
                img = np.ravel(img)
                x.append(img)
                y.append(1)
    for filename in os.listdir(y_dir):
        f = os.path.join(y_dir, filename)
        if filename != 'desktop.ini':
            if os.path.isfile(f):
                img = Image.open(f).convert('L')
                img = np.array(img)
                img = np.ravel(img)
                x.append(img)
                y.append(-1)
    x = np.array(x)
    return x, y