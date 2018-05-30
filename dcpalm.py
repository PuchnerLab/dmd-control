#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import re
import yaml

from matplotlib.path import Path


def make_path(roi_json):
    return Path(list(zip(roi_json['x'], roi_json['y'])))


def load_rois(roi_json_file):
    with open(roi_json_file, 'r') as f:
        rois = json.load(f)[1]
    return rois


def load_mol_list(mol_list_file):
    fields = [
        'x', 'y', 'xc', 'yc', 'h', 'a', 'w', 'phi', 'ax', 'bg', 'i', 'c', 'fi',
        'fr', 'tl', 'lk', 'z', 'zc'
    ]
    ext = mol_list_file.split('.')[-1]
    if ext == 'bin':
        from storm_analysis.sa_library.readinsight3 import I3Reader
        mol_list_bin = I3Reader(mol_list_file)
        mol_list = pd.DataFrame(mol_list_bin.localizations, columns=fields)
    else:
        mol_list = pd.read_csv(mol_list_file, sep='\t')
    return mol_list


def in_roi(mol_list, roi_path):
    return mol_list[roi_path.contains_points(mol_list[['x', 'y']])]


def estimate_roi_path_area(roi_path, size=(256, 256)):
    return roi_path.contains_points([[i, j] for i in range(size[0]) for j in range(size[1])]).sum()


def count_mol_in_roi(mol_list, roi_path):
    return roi_path.contains_points(mol_list[['x', 'y']]).sum()

def plot_mol_list(mol_list, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # title_string += '{r}: N = {n}   A = {a}\n'.format(
    #     r=r, n=mol_list.shape[0], a=area)

    ax.scatter(
        mol_list['x'],
        mol_list['y'],
        c=set_alpha(mol_list['h']), # , color),
        s=10 * mol_list['w'] / mol_list['w'].max(),
        edgecolor='none',
        **kwargs)
    # plt.hist2d(mol_list['x'], mol_list['y'],
    #            bins=path.vertices.max(axis=0)-path.vertices.min(axis=0))
    # plt.plot(roi['x'], roi['y'], c=color)  # , label=r

    return fig, ax


def plot_roi(roi, ax, **kwargs):
    # color = ax._get_lines.get_next_color()
    ax.plot(roi['x'], roi['y'], **kwargs)


def set_alpha(arr, color=mpl.colors.TABLEAU_COLORS['tab:blue']):
    colors = np.zeros((arr.size, 4))
    colors[:, 3] = arr / arr.max()
    colors[:, 0:3] = mpl.colors.hex2color(color)
    return colors


def set_labels(fig_title, fig, ax):
    fig.legend()
    fig.suptitle(fig_title)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')


def set_style(fig, ax):
    ax.set_aspect(1)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.invert_yaxis()


def plot_full(mol_list, **kwargs):
    fig, ax = plot_mol_list(mol_list, **kwargs)
    set_labels(fig, ax)
    set_style(fig, ax)
    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_spec', metavar='analysis-spec',
                        help='yaml file')
    parser.add_argument('--lag-time', action='store_true',
                        default=False)
    parser.add_argument('--saveas', default='')
    args = parser.parse_args()

    # This order is better for plotting.
    roi_names = ['cell', 'cytoplasm']
    # roi_names = ['view', 'cell', 'cytoplasm']
    data = dict()

    plt.close('all')

    print(args)
    # Open yaml file containing movies to analysize and regions of interest.
    analysis_spec = yaml.load(open(args.analysis_spec, 'r'))
    for movie, params in analysis_spec.items():
        mol_list = load_mol_list(params['mlist'])
        # plt.scatter(mol_list['x'], mol_list['y'],
        #             c=gen_alpha(mol_list['h']),
        #             s=10*mol_list['w']/mol_list['w'].max())
        # plt.hist2d(mol_list['x'], mol_list['y'],
        #            bins=256, cmax=300)
        # plt.imshow(plt.imread(params['average'], 'grayscale'), cmap='gray')
        # fig, ax = plot_mol_list(mol_list)
        title_str = str(movie)
        if args.lag_time:
            title_str += ', lag time: ' + str(params['lag-time'])

        fig, ax = plot_mol_list(mol_list, label=None)
        for cell_num, roi_json_file in enumerate(params['rois']):
            rois = load_rois(roi_json_file)
            for roi_name, roi in rois.items():
                if roi_name != 'view':
                    roi_path = make_path(roi)
                    plot_roi(roi, ax, label='{}: Count: {}; Area: {}'.format(
                        roi_name,
                        count_mol_in_roi(mol_list, roi_path),
                        estimate_roi_path_area(roi_path)
                    ))
        set_labels(title_str, fig, ax)
        set_style(fig, ax)

        if args.saveas:
            plt.savefig('dcpalm_figures/' +
                        re.sub(r'(.*)(list?)\..*$',
                               r'\1dcpalm.' + args.saveas,
                               params['mlist']))
            # params['mlist'].replace(
            #     'list.bin', 'dcpalm.' + args.saveas)
    plt.show()
