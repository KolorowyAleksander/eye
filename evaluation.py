#!/usr/bin/env python3
'''
Evaluates two masks, optionally showing them on a result
'''
from datetime import datetime
from argparse import ArgumentParser
from math import sqrt

from cv2 import imread, imwrite, add
from numpy import array_equal, hstack, vstack

from utils import existing_file, WHITE, BLACK


def compare_images(original, tested):
    '''Compares two binary mask, original and measured'''
    matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for row1, row2 in zip(original, tested):
        for e1, e2 in zip(row1, row2):
            if array_equal(e1, WHITE) and array_equal(e2, WHITE):
                matrix['tp'] += 1
            elif array_equal(e1, WHITE) and array_equal(e2, BLACK):
                matrix['fn'] += 1
            elif array_equal(e1, BLACK) and array_equal(e2, WHITE):
                matrix['fp'] += 1
            elif array_equal(e1, BLACK) and array_equal(e2, BLACK):
                matrix['tn'] += 1

    measures = {'acc': (matrix['tp'] + matrix['tn'])/sum(matrix.values()),
                'prec': matrix['tp']/(matrix['tp'] + matrix['fp']),
                'sen': matrix['tp']/(matrix['tp'] + matrix['fn']),
                'spec': matrix['tn']/(matrix['tn'] + matrix['fp'])}
    measures['g-mean'] = sqrt(measures['sen'] * measures['prec'])

    return measures, matrix


def main():
    parser = ArgumentParser()
    parser.add_argument('mask', help='original mask', type=existing_file)
    parser.add_argument('test', help='measured mask', type=existing_file)
    parser.add_argument('--draw', help='draw comparison', type=existing_file)
    args = parser.parse_args()

    mask = imread(args.mask)
    test = imread(args.test)

    if args.draw is not None:
        img = imread(args.draw)

    if mask.shape != test.shape or img is not None and img.shape != mask.shape:
        exit('Images dimensions differ {}/{}'.format(mask.shape, test.shape)
             + img.shape if img is not None else '')

    stats, measures = compare_images(mask, test)

    if img is not None:
        cmp = draw_comparison(img, mask, test)
        imwrite('bin/comparison' + datetime.now().isoformat('T') + '.jpg', cmp)

    print(stats)
    print(measures)


def draw_comparison(img, mask, test):
    return vstack((hstack((img, add(img, test))), hstack((mask, test))))


if __name__ == '__main__':
    main()
