#!/usr/bin/env python3
'''
Try predicting the image with traditional file processing methods
'''
from argparse import ArgumentParser
from datetime import datetime

from cv2 import (createCLAHE, cvtColor, GaussianBlur, morphologyEx, imread,
                 imwrite, adaptiveThreshold, COLOR_BGR2GRAY, MORPH_OPEN,
                 ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY)
import numpy as np

from utils import existing_file

KERNEL = np.ones((5, 5), np.uint8)
GAUSS_SIGMA = 5
GAUSS_SIZE = (7, 7)
clahe = createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))


def process_image(image):
    '''Standard CLAHE/adaptive thresholding'''
    image = cvtColor(image, COLOR_BGR2GRAY)
    image = GaussianBlur(image, GAUSS_SIZE, GAUSS_SIGMA)

    image = clahe.apply(image)
    image = morphologyEx(image, MORPH_OPEN, KERNEL, iterations=1)
    image = adaptiveThreshold(image,
                              255,
                              ADAPTIVE_THRESH_GAUSSIAN_C,
                              THRESH_BINARY,
                              17,
                              2)
    return image


def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='original image', type=existing_file)
    parser.add_argument('mask', help='original mask', type=existing_file)
    args = parser.parse_args()

    image = imread(args.image)
    image = process_image(image)

    imwrite('bin/processed' + datetime.now().isoformat('T') + '.png', image)


if __name__ == '__main__':
    main()
