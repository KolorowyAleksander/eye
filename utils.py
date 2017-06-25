from argparse import ArgumentTypeError
from os.path import exists, isdir

from cv2 import copyMakeBorder, BORDER_CONSTANT
import numpy as np

BLACK = np.array([0, 0, 0])
WHITE = np.array([255, 255, 255])


def add_black_border(image, pixels):
    return copyMakeBorder(image,
                          pixels,
                          pixels,
                          pixels,
                          pixels,
                          BORDER_CONSTANT,
                          value=[0, 0, 0])


def existing_file(filename):
    if not exists(filename):
        raise ArgumentTypeError('{} is not an existing file'.format(filename))
    return filename


def existing_directory(dirname):
    if not exists(dirname) or not isdir(dirname):
        raise ArgumentTypeError('{} directory doesn\'t exist'.format(dirname))
    return dirname
