#!/usr/bin/env python3
'''
Predicts a mask from image given a model
'''
from sys import argv
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
from cv2 import imread, imwrite
from keras.models import model_from_json

from utils import add_black_border, existing_file

S_LEN = 21


def load_model(model_name):
    with open('bin/{}.json'.format(model_name), 'r') as file:
        model = model_from_json(file.read())
        model.load_weights('bin/{}.h5'.format(model_name))
    return model


def predict(image, model_name):
    model = load_model(model_name)

    original_height = len(image)
    original_width = len(image[0])

    image = add_black_border(image, S_LEN//2)

    image = np.array(image, dtype=np.float32)
    image /= 255

    samples = []
    for i in range(original_height):
        for j in range(original_width):
            samples.append(image[i:i+S_LEN, j:j+S_LEN])

    samples = np.array(samples)
    result = model.predict(samples, batch_size=32, verbose=1)
    result = np.array([0 if x[0] > x[1] else 255 for x in result])

    return np.reshape(result, (original_height, original_width))


def main(args):
    parser = ArgumentParser()
    parser.add_argument('image', help='predicted image', type=existing_file)
    parser.add_argument('model', help='model name')

    args = parser.parse_args()
    image = imread(args.image)
    result = predict(image, args.model)

    imwrite('bin/result' + datetime.now().isoformat('T') + '.png', result)


if __name__ == '__main__':
    main(argv)
