#!/usr/bin/env python3
from os import sep, listdir
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import cv2 as cv

from model import model
from utils import existing_directory

batch_size = 128
epochs = 10

model = model()


def main():
    parser = ArgumentParser()
    parser.add_argument('sources', help='sources', type=existing_directory)
    parser.add_argument('--model',
                        help='model name',
                        default='model' + datetime.now().isoformat('T'))
    args = parser.parse_args()

    k = 2
    images, labels = load_images_and_labels(args.sources)

    train_data = images[len(images)//k:]
    train_labels = labels[len(images)//k:]

    test_data = images[:len(images)//k]
    test_labels = labels[:len(labels)//k]

    model.fit(train_data,
              train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(test_data, test_labels))

    with open('bin/{}.json'.format(args.model), 'w+') as file:
        file.write(model.to_json())

    model.save_weights('bin/{}.h5'.format(args.model))


def load_images_and_labels(directory):
    names = sorted(listdir(directory))

    images = []
    labels = []

    for name in names:
        image = cv.imread(directory + sep + name)
        label = np.array([1, 0] if name[8] == '0' else [0, 1])
        images.append(image)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    images /= 255  # conversion to 0.0-1.0 float

    return images, np.array(labels)


if __name__ == '__main__':
    main()
