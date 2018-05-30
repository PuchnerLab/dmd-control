import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import feature, io, measure, morphology


"""This code can be used to find a spot from the camera and instead of transforming the entire image, we can find the centroid of the spot, transform and """


def label_image(img, min_area=90, min_circ=0.25):
    """Applies a Canny feature detection to detect edges on an image `img` and filters the edges based on area and circularity.
    
    Parameters
    ----------
    img : Image containing features to be detected

    min_area : Minimum area of the labels to be retained in square pixels, optional

    min_circ : Minimum circularity of the labels to be retained, optional

    Returns
    -------
    out : List of `skimage.measure._regionprops._RegionProperties` objects

    """

    edges = feature.canny(img, sigma=3, low_threshold=0.08, high_threshold=0.11)
    label_img = morphology.label(edges)
    good_labels = [l for l in measure.regionprops(label_img) if
                   l.area > min_area and circularity(l) > min_circ]
    return good_labels


def plot_labels(fig, ax, img, **kwargs):
    ax.imshow(img)
    for region in label_image(img):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr),
                                  maxc - minc,
                                  maxr - minr,
                                  fill=False,
                                  edgecolor='red',
                                  linewidth=2,
                                  **kwargs)
        ax.add_patch(rect)


def plot_centroids(fig, ax, labels, **kwargs):
    centroids = [l.centroid for l in labels]
    ax.scatter(*list(zip(*centroids))[-1::-1],
               **kwargs)


def circularity(labeled_edge):
    return labeled_edge.area / (labeled_edge.perimeter / (2 * np.pi))**2
