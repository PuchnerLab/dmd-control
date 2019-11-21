# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:02:53 2017

@author: Joe Eix & Angel Mancebo

    Takes a source image from DMA and the destination image from the
    image plane and finds the transformation matrix. Calibration
    images consist of four circles, with the centers being found
    through blob detection.
"""

from __future__ import division, print_function

import sys
from argparse import ArgumentParser

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import yaml
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import feature, io, measure, morphology, transform


def detect_points(img, scale=1):
    img_copy = (img * ((2**8 - 1) / img.max()))[:].astype('uint8')

    params = cv2.SimpleBlobDetector_Params()

    params.thresholdStep = 1
    params.minThreshold = 70
    params.maxThreshold = 255

    params.filterByColor = False

    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100

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

    .. math:: G(x, y; a, b, c) = \\exp(-(a x^2 + b x y + c y^2))

    to each blob within a window of the size of the blob size. This
    code leverages the elliptical Gaussian fitting functions from
    storm_analysis by Hazen Babcock[1].

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
        x, y, r = blobs[b, :].astype('uint16')
        r *= 2
        # image row-col convention is the opposite of x-y
        region = img[y - r:y + r, x - r:x + r]
        params = fitEllipticalGaussian(region)
        gaussians[b, :] = params[0][2:7]
        gaussians[b, 0:2] += blobs[b, 0:2] - r
        print(params)
    return gaussians


def label_image(img, min_area=10, min_circ=0.25):
    """Applies a Canny feature detection to detect edges on an image
    `img` and filters the edges based on area and circularity.

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
    img = io.imread(img_file, as_gray=True)
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


def sort_points(pts, invertx=False, inverty=False):
    # Sort vertically
    xmin = np.max if invertx else np.min
    xmax = np.min if invertx else np.max
    ymin = np.max if inverty else np.min
    ymax = np.min if inverty else np.max
    x, y = np.mgrid[
        int(xmin(pts[:, 0])):int(xmax(pts[:, 0])):(-1 if invertx else 1),
        int(ymin(pts[:, 1])):int(ymax(pts[:, 1])):(-1 if inverty else 1)]
    tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))
    distances, indices = tree.query(np.array(pts[:, :2]))
    return pts[np.argsort(indices)]


def main():
    parser = ArgumentParser(description='Process stuff.')

    parser.add_argument(
        '--screen',
        type=str,
        default='',
        help="""
             Path of image of calibration image on as it appears on
             screen.
             """)

    parser.add_argument(
        '--sample',
        type=str,
        default='',
        help="""
             Path of image of calibration image on sample.
             """)

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help="""
             Path of file to output the column-separated
             transformation matrix.
             """)

    parser.add_argument(
        '--fit',
        type=int,
        default=1,
        help="""
             Perform Gaussian fitting after blob detection. Default is
             1 (perform fitting). Supply 0 to skip fitting.
             """)

    parser.add_argument(
        '--invertx',
        type=bool,
        default=False,
        help="""
             Inidicate whether the camera image is y-inverted. This
             will be taken into account in the transfomation by
             forcing a reversal of the sorted keypoints.
             """)

    parser.add_argument(
        '--inverty',
        type=bool,
        default=True,
        help="""
             Inidicate whether the camera image is y-inverted. This
             will be taken into account in the transfomation by
             forcing a reversal of the sorted keypoints.
             """)

    parser.add_argument(
        '--params',
        type=str,
        default=None,
        help="""
             Path of parameters file. Options set in this file
             override all other options. See example parameters file
             for more information.
             """)

    args, _ = parser.parse_known_args()
    print(sys.argv)
    if not args.screen:
        print('Provide a screen image.', file=sys.stderr)
        sys.exit(1)
    if not args.sample:
        print('Provide a sample image.', file=sys.stderr)
        sys.exit(1)

    with open(args.params, 'r') as params_file:
        params = yaml.load(params_file)

    screen = load_image(args.screen)
    sample = load_image(args.sample)
    if params['camerascale'] != 1:
        sample = transform.rescale(sample, params['camerascale'])

    screen_coords = detect_points(screen)
    sample_coords = detect_points(sample)

    screen_coords_sorted = sort_points(screen_coords, params['invertx'],
                                       params['inverty'])
    sample_coords_sorted = sort_points(sample_coords)

    if args.fit:
        sample_coords_sorted = fit_gaussian(sample, sample_coords_sorted)

    tparams = {'ttype': 'polynomial', 'order': 3}
    tform = transform.estimate_transform(
        src=screen_coords_sorted, dst=sample_coords_sorted, **tparams)
    tform_inv = transform.estimate_transform(
        src=sample_coords_sorted, dst=screen_coords_sorted, **tparams)

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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    ax[0, 0].imshow(sample, cmap=plt.cm.gray)
    ax[0, 0].plot(
        sample_coords_sorted[:, 0],
        sample_coords_sorted[:, 1],
        color='tab:red',
        marker='o',
        markersize=3,
        linewidth=1)
    ax[0, 0].plot(
        tform(screen_coords_sorted)[:, 0],
        tform(screen_coords_sorted)[:, 1],
        color='tab:blue',
        marker='o',
        markersize=3,
        linewidth=1)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    scalebar = ScaleBar(
        params['pixelsize_nm'] * 1e-9 / params['camerascale'],
        frameon=False,
        color='white',
        location='lower right')
    ax[0, 0].add_artist(scalebar)

    ax[0, 1].imshow(screen, cmap=plt.cm.gray)
    ax[0, 1].plot(
        screen_coords_sorted[:, 0],
        screen_coords_sorted[:, 1],
        color='tab:red',
        marker='o',
        markersize=3,
        linewidth=1)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[1, 0].imshow(screen_tform, cmap=plt.cm.gray)
    ax[1, 0].plot(
        tform(screen_coords_sorted)[:, 0],
        tform(screen_coords_sorted)[:, 1],
        color='tab:red',
        marker='o',
        markersize=3,
        linewidth=1)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    def vec_length(array):
        return np.sqrt(np.sum(array**2, 1))

    distance = vec_length(sample_coords_sorted[:, :2] -
                          tform(screen_coords_sorted[:, :2]))
    ax[1, 1].violinplot(
        params['pixelsize_nm'] * distance / params['camerascale'],
        widths=0.25,
        showmeans=True,
        showmedians=True)
    ax[1, 1].grid(axis='y')
    ax[1, 1].set_xlim((0, 2))
    ax[1, 1].set_xticks([])
    ax[1, 1].set_ylabel('discrepancy (nm)')
    ax[1, 1].legend()
    fig.tight_layout()
    plt.show()

    return tform.params


if __name__ == '__main__':
    main()
