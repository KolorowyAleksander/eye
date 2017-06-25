'''
Takes all images in IMAGES and makes 21x21 samples in IMAGES_DEST
Uses labels from LABELS, taking center pixel from those
'''
from os import listdir, sep
from random import randrange
from argparse import ArgumentParser

from cv2 import imread, imwrite, IMREAD_GRAYSCALE

from utils import existing_directory

FMT = '{}' + sep + '{:07d}_{}.png'
R2 = 21//2  # image radius is 21


def main():
    parser = ArgumentParser()
    parser.add_argument('images', help='sources', type=existing_directory)
    parser.add_argument('labels', help='labels', type=existing_directory)
    parser.add_argument('dest', help='destination', type=existing_directory)
    parser.add_argument('--number', '-n', type=int, default=10000,
                        help='~~ number of samples')
    args = parser.parse_args()

    images = args.images
    labels = args.labels
    dest = args.dest
    counter = 0

    number_for_each = args.number//len(listdir(images))

    for f in listdir(images):
        image = imread(images + sep + f)
        label = imread(labels + sep + f, IMREAD_GRAYSCALE)

        w = len(image)
        h = len(image[0])

        for j in range(0, number_for_each):
            x = randrange(R2, w-R2)
            y = randrange(R2, h-R2)

            sample = image[x-R2:x+R2+1, y-R2:y+R2+1]
            l = label[x, y]//255

            imwrite(FMT.format(dest, counter, l), sample)

            counter += 1


if __name__ == '__main__':
    main()
