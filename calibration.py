# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:02:53 2017

@author: Joe Eix & Angel Mancebo

Takes a source image from DMA and the destination image from the image plane
and finds the transformation matrix. Calibration images consist of four circles,
with the centers being found through blob detection.
"""

from __future__ import division, print_function

import sys
from argparse import ArgumentParser

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature, io, measure, morphology, transform

import cv2


def detect_points(img, scale=1):
    img_copy = (img * ((2**8 - 1) / img.max()))[:].astype('uint8')

    params = cv2.SimpleBlobDetector_Params()

    params.thresholdStep = 1
    params.minThreshold = 100
    params.maxThreshold = 255

    params.filterByColor = False

    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 1000

    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1.0

    params.filterByInertia = False

    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_copy)
    # img_with_keypoints = cv2.drawKeypoints(img_copy, keypoints, np.array([]),
    #                                        (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return np.array([list(k.pt) + [k.size] for k in keypoints])
    # return img_with_keypoints


def fit_gaussian(img, blobs):
    """After blob detection, fit an elliptical Gaussian of the form

    .. math:: G(x, y; a, b, c) = \\exp(a x^2 + b x y + c y^2)

    to each blob within a window of the size of the blob size. This code leverages the elliptical Gaussian fitting functions from storm_analysis by Hazen Babcock[1].

    Parameters
    ----------
    img : ndarray
        Image on which the blob detection was performed.

    blobs : ndarray
        Array of blobs of the form [x, y, size].

    Returns
    -------
    out : ndarray
        Array of Gaussian parameters of the form [x, y, a, b, c].

    References
    ----------
    .. [1] https://github.com/ZhuangLab/storm-analysis/blob/master/storm_analysis/sa_library/gaussfit.py
    """
    from storm_analysis.sa_library.gaussfit import fitEllipticalGaussian
    gaussians = np.zeros((blobs.shape[0], 5), dtype=blobs.dtype)
    for b, _ in enumerate(blobs):
        radius = 2 * blobs[b, 2]
        region = img[int(blobs[b, 0] - radius):int(blobs[b, 0] + radius),
                     int(blobs[b, 1] - radius):int(blobs[b, 1] + radius)]
        params = fitEllipticalGaussian(region)
        gaussians[b, :] = params[0][2:7]
        gaussians[b, 0:2] += blobs[b, 0:2] - radius
        print(params)
    return gaussians


def label_image(img, min_area=90, min_circ=0.25):
    """Applies a Canny feature detection to detect edges on an image `img` and filters the edges based on area and circularity.

    Parameters
    ----------
    img : ndarray
        Image containing features to be detected

    min_area : scalar, optional
        Minimum area of the labels to be retained in square pixels

    min_circ : scalar, optional
        Minimum circularity of the labels to be retained

    Returns
    -------
    out : array_like
        List of `skimage.measure._regionprops._RegionProperties` objects

    """

    edges = feature.canny(
        img, sigma=3, low_threshold=0.08, high_threshold=0.11)
    label_img = morphology.label(edges)
    good_labels = [
        l for l in measure.regionprops(label_img)
        if l.area > min_area and circularity(l) > min_circ
    ]
    return good_labels


def load_image(img_file):
    img = io.imread(img_file, as_grey=True)
    return img


def plot_labels(fig, ax, img, **kwargs):
    ax.imshow(img)
    for region in label_image(img):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=2,
            **kwargs)
        ax.add_patch(rect)


def plot_centroids(fig, ax, labels, **kwargs):
    centroids = [l.centroid for l in labels]
    ax.scatter(*list(zip(*centroids))[-1::-1], **kwargs)


def circularity(labeled_edge):
    return labeled_edge.area / (labeled_edge.perimeter / (2 * np.pi))**2


def sort_points(points, npoints=(4, 4)):
    """Sorts a grid pattern. The default is a 4x4 grid.
    """
    width, height = npoints
    tmp = points.copy()

    # Construct view with same datatype as ``points`` to select either
    # column or row to sort by.
    stack = []
    viewtype = '{0},{0},{0}'.format(points.dtype)
    # Sort in-place
    tmp.view(viewtype).sort(order=['f1'], axis=0)
    for j in range(height):
        stack.append(tmp[height * j:height * (j + 1), :])
    for s in stack:
        s.view(viewtype).sort(order=['f0'], axis=0)
    return np.vstack(stack)


if __name__ == '__main__':
    parser = ArgumentParser(description='Process stuff.')

    parser.add_argument(
        '--screen',
        type=str,
        default='',
        help="""Path of image of calibration image on
                             as it appears on screen.
                             """)

    parser.add_argument(
        '--sample',
        type=str,
        default='',
        help="""Path of image of calibration image on
                             sample.
                             """)

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help="""Path of file to output the
                             column-separated transformation matrix.
                             """)

    parser.add_argument(
        '--npoints',
        nargs='+',
        type=int,
        # TODO Add help for npoints, 3 or 4 arguments
        # depending on calibration pattern.
        help="""""")

    parser.add_argument(
        '--ttype',
        type=str,
        default='polynomial',
        help="""The type of transformation to use.
                             Allowed values are polynomial and
                             projective. By default a polynomial of
                             order 3 is used. Polynomial should be
                             more accurate, but if the transformation
                             appears unstable, then use projective.
                             """)
    # help="""Triple of number of points in the Z
    #      shape of the calibration image. Default
    #      is '3 2 3': 3 points on top, 2 points
    #      diagonally down, and 3 points on the
    #      bottom.
    #      """

    parser.add_argument(
        '--fit',
        type=int,
        default=1,
        help="""Perform Gaussian fitting after blob detection.
                             default is 1 (perform fitting).  Supply 0 to skip
                             fitting..
                             """)

    args, _ = parser.parse_known_args()

    if not args.screen:
        print('Provide a screen image.', file=sys.stderr)
        sys.exit(1)
    if not args.sample:
        print('Provide a sample image.', file=sys.stderr)
        sys.exit(1)

    # screen = '../dma_calibration_image_Z_20171103.png'
    # sample = '../movie_0002_calibration_box_smudge.png'

    screen = load_image(args.screen)
    sample = load_image(args.sample)
    if sample.shape != (512, 512):
        sample = transform.resize(sample, (512, 512))

    screen_coords = detect_points(screen)
    # screen_coords[:, :2] = screen_coords[:, 1::-1]

    sample_coords = detect_points(sample)
    # sample_coords[:, :2] = sample_coords[:, 1::-1]

    screen_coords_sorted = sort_points(screen_coords, args.npoints)
    # Account for 180 degree rotation by reversing order of points
    sample_coords_sorted = sort_points(sample_coords, args.npoints[-1::-1])

    if args.fit:
        sample_coords_sorted[:, :2] = sample_coords_sorted[:, 1::-1]
        sample_coords_sorted = fit_gaussian(sample, sample_coords_sorted)
        sample_coords_sorted[:, :2] = sample_coords_sorted[:, 1::-1]
    sample_coords_sorted[:] = sample_coords_sorted[-1::-1]

    # # Transformation of screen -> sample
    # tform = transform.estimate_transform(ttype='projective',
    #                               src=screen_coords_sorted,
    #                               dst=sample_coords_sorted)

    tparams = {'ttype': args.ttype}
    if args.ttype == 'polynomial':
        tparams['order'] = 3

    tform = transform.estimate_transform(
        src=screen_coords_sorted, dst=sample_coords_sorted, **tparams)

    tform_inv = transform.estimate_transform(
        src=sample_coords_sorted, dst=screen_coords_sorted, **tparams)

    sample_tform = transform.warp(
        image=sample.copy(),
        inverse_map=tform,
        output_shape=screen.shape,
        order=0)

    # Apply inverse transformation to screen. After this passes
    # through, the optical path, it will be trasformed, resulting in
    # overall identity transformation.
    screen_tform = transform.warp(
        image=screen.copy(),
        inverse_map=tform_inv,
        output_shape=sample.shape,
        order=0)

    if args.output != '':
        np.savetxt(args.output, tform.params, delimiter=',')
    else:
        print(tform.params)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    ax[0, 0].imshow(sample, cmap=plt.cm.gray)
    ax[0, 0].plot(
        sample_coords_sorted[:, 0],
        sample_coords_sorted[:, 1],
        '-or',
        markersize=3,
        linewidth=1)
    ax[0, 0].plot(
        tform(screen_coords_sorted)[:, 0],
        tform(screen_coords_sorted)[:, 1],
        '-ob',
        markersize=3,
        linewidth=1)
    ax[0, 0].set_xlabel('x (px)')
    ax[0, 0].set_ylabel('y (px)')

    ax[0, 1].imshow(screen, cmap=plt.cm.gray)
    ax[0, 1].plot(
        screen_coords_sorted[:, 0],
        screen_coords_sorted[:, 1],
        '-or',
        markersize=3,
        linewidth=1)
    ax[0, 1].set_xlabel('x (px)')
    ax[0, 1].set_ylabel('y (px)')

    ax[1, 0].imshow(screen_tform, cmap=plt.cm.gray)
    ax[1, 0].plot(
        tform(screen_coords_sorted)[:, 0],
        tform(screen_coords_sorted)[:, 1],
        '-or',
        markersize=3,
        linewidth=1)
    ax[1, 0].set_xlabel('x (px)')
    ax[1, 0].set_ylabel('y (px)')

    def vec_length(array):
        return np.sqrt(np.sum(array**2, 1))

    distance = vec_length(sample_coords_sorted[:, :2] -
                          tform(screen_coords_sorted[:, :2]))
    ax[1, 1].plot(
        distance, label='{:0.4f} +/- {:0.4f} nm'.format(160 * distance.mean() / 2, 160 * distance.std() / 2))
    ax[1, 1].set_xlabel('point')
    ax[1, 1].set_ylabel('discrepancy (nm)')
    ax[1, 1].legend()
    fig.tight_layout()
    plt.show()
