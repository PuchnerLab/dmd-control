# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:02:53 2017

@author: Joe Eix & Angel Mancebo

Takes a source image from DMA and the destination image from the image plane
and finds the transformation matrix. Calibration images consist of four circles,
with the centers being found through blob detection. 
"""

from __future__ import print_function

import cv2
import numpy as np
import sys
from skimage import exposure
from skimage import feature
from skimage import filters
from skimage import io
from skimage import transform as tf


def load_image(img_file):
    # 0 flag for loading as grayscale
    # img = cv2.imread(img_file, 0)
    img = io.imread(img_file, as_grey=True)
    return img


def detect_points(img, scale=1):
    from starfish.filters import white_top_hat
    min_sigma = scale * 10
    max_sigma = scale * 30
    num_sigma = (max_sigma - min_sigma) // scale

    img_top_hat = white_top_hat(img.astype('float64'), min_sigma)

    threshold = filters.threshold_yen(img_top_hat)

    blobs = feature.blob.blob_log(img_top_hat,
                                  min_sigma=min_sigma,
                                  max_sigma=max_sigma,
                                  num_sigma=num_sigma,
                                  threshold=threshold)

    coords = blobs[:, 1::-1]
    return coords


def apply_transform(src, m):
    src3d = np.hstack((src, np.zeros((src.shape[0], 1))))
    src3d_transf = np.apply_along_axis(lambda x: np.dot(m, x),
                                       1,
                                       src3d)
    src_transf = src3d_transf[:, :2]
    return np.array(src_transf, dtype=src.dtype)


def sort_points(points, npoints):
    if len(npoints) == 3:
        return sort_points_z(points, npoints)
    if len(npoints) == 2:
        return sort_points_grid(points, npoints)
    else:
        print('Invalid ``npoints`` dimension', file=sys.stderr)


def sort_points_z(points, npoints=(3, 2, 3)):
    """Sorts an upright 'Z' pattern. The default is an upright 'Z' of
    8 points, with 3 on top, 2 diagonally in the middle, and 3 on the
    bottom.
    """
    ntop, nmid, nbot = npoints
    tmp = points.copy()

    # Construct view with same datatype as ``points`` to select either
    # column or row to sort by.
    viewtype = '{0}, {0}'.format(points.dtype)
    # Sort in-place
    tmp.view(viewtype).sort(order=['f1'], axis=0)
    top = tmp[:ntop, :]
    top.view(viewtype).sort(order=['f0'], axis=0)
    bot = tmp[-nbot:, :]
    bot.view(viewtype).sort(order=['f0'], axis=0)
    mid = tmp[ntop:ntop+nmid]
    return np.vstack((top, mid, bot))


def sort_points_grid(points, npoints=(4, 4)):
    """Sorts a grid pattern. The default is a 4x4 grid.
    """
    width, height = npoints
    tmp = points.copy()

    # Construct view with same datatype as ``points`` to select either
    # column or row to sort by.
    stack = []
    viewtype = '{0},{0}'.format(points.dtype)
    # Sort in-place
    tmp.view(viewtype).sort(order=['f1'], axis=0)
    for j in range(height):
        stack.append(tmp[height*j:height*(j+1), :])
    for s in stack:
        s.view(viewtype).sort(order=['f0'], axis=0)
    return np.vstack(stack)


def transform(src_image, src_to_dst_transform, dst_dim):
    """Returns transformed image array from an original source image
    and a given transformation matrix.
    """
    return cv2.warpPerspective(src_image, src_to_dst_transform, dst_dim)
