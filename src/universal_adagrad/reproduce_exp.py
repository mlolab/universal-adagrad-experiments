import argparse
import os

import matplotlib.pyplot as plt
import pooch
from haven import haven_wizard as hw

from universal_adagrad import exp_configs
from universal_adagrad.plot_helper import plot_result
from universal_adagrad.trainval import trainval


def form_styles_polyhedron(linewidth, markevery, markersize):
    styles = {}

    styles["UniSgd-AdaGrad"] = {
        "label": "UniSgd (ours)",
        "color": "black",
        "marker": "o",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd-AdaGrad"],
    }

    styles["UniSvrg-AdaGrad"] = {
        "label": "UniSvrg (ours)",
        "color": "blue",
        "marker": "v",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg-AdaGrad"],
    }

    styles["UniFastSgd-AdaGrad"] = {
        "label": "UniFastSgd (ours)",
        "color": "steelblue",
        "marker": "^",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd-AdaGrad"],
    }

    styles["UniFastSvrg-AdaGrad"] = {
        "label": "UniFastSvrg (ours)",
        "color": "darkorchid",
        "marker": "*",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg-AdaGrad"],
    }

    styles["AdaSVRG"] = {
        "label": "AdaSVRG",
        "color": "green",
        "marker": ">",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["AdaSVRG"],
    }

    styles["AdaVRAE"] = {
        "label": "AdaVRAE",
        "color": "gray",
        "marker": "<",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["AdaVRAE"],
    }

    styles["AdaVRAG"] = {
        "label": "AdaVRAG",
        "color": "darkorange",
        "marker": "p",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["AdaVRAG"],
    }

    styles["AccSVRG"] = {
        "label": "FastSvrg",
        "color": "fuchsia",
        "marker": "x",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["AccSVRG"],
    }

    return styles


def form_styles_minibatch(linewidth, markevery, markersize):
    styles = {}

    styles["UniFastSvrg-AdaGrad-B-64"] = {
        "label": "batchsize=64",
        "color": "blue",
        "marker": "^",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg-64"],
    }

    styles["UniFastSvrg-AdaGrad-B-512"] = {
        "label": "batchsize=512",
        "color": "red",
        "marker": "^",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg-512"],
    }

    styles["UniFastSvrg-AdaGrad-B-4096"] = {
        "label": "batchsize=4096",
        "color": "orange",
        "marker": "^",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg-4096"],
    }

    styles["UniSvrg-AdaGrad-B-64"] = {
        "label": "batchsize=64",
        "color": "blue",
        "marker": "o",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg-64"],
    }

    styles["UniSvrg-AdaGrad-B-512"] = {
        "label": "batchsize=512",
        "color": "red",
        "marker": "o",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg-512"],
    }

    styles["UniSvrg-AdaGrad-B-4096"] = {
        "label": "batchsize=4096",
        "color": "orange",
        "marker": "o",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg-4096"],
    }

    styles["UniFastSgd-AdaGrad-B-64"] = {
        "label": "batchsize=64",
        "color": "blue",
        "marker": "v",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd-64"],
    }

    styles["UniFastSgd-AdaGrad-B-512"] = {
        "label": "batchsize=512",
        "color": "red",
        "marker": "v",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd-512"],
    }

    styles["UniFastSgd-AdaGrad-B-4096"] = {
        "label": "batchsize=4096",
        "color": "orange",
        "marker": "v",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd-4096"],
    }

    styles["UniSgd-AdaGrad-B-64"] = {
        "label": "batchsize=64",
        "color": "blue",
        "marker": "x",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd-64"],
    }

    styles["UniSgd-AdaGrad-B-512"] = {
        "label": "batchsize=512",
        "color": "red",
        "marker": "x",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd-512"],
    }

    styles["UniSgd-AdaGrad-B-4096"] = {
        "label": "batchsize=4096",
        "color": "orange",
        "marker": "x",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd-4096"],
    }

    return styles


def form_styles_comp(linewidth, markevery, markersize):
    styles = {}

    styles["UniSgd-AdaGrad"] = {
        "label": "UniSgd-AdaGrad",
        "color": "black",
        "marker": "^",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd-AdaGrad"],
    }

    styles["UniSvrg-AdaGrad"] = {
        "label": "UniSvrg-AdaGrad",
        "color": "blue",
        "marker": "v",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg-AdaGrad"],
    }

    styles["UniFastSgd-AdaGrad"] = {
        "label": "UniFastSgd-AdaGrad",
        "color": "steelblue",
        "marker": "o",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd-AdaGrad"],
    }

    styles["UniFastSvrg-AdaGrad"] = {
        "label": "UniFastSvrg-AdaGrad",
        "color": "darkorchid",
        "marker": "*",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg-AdaGrad"],
    }

    styles["UniSgd"] = {
        "label": "UniSgd-Other",
        "color": "green",
        "marker": ">",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSgd"],
    }

    styles["UniSvrg"] = {
        "label": "UniSvrg-Other",
        "color": "gray",
        "marker": "<",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniSvrg"],
    }

    styles["UniFastSgd"] = {
        "label": "UniFastSgd-Other",
        "color": "darkorange",
        "marker": "p",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSgd"],
    }

    styles["UniFastSvrg"] = {
        "label": "UniFastSvrg-Other",
        "color": "fuchsia",
        "marker": "x",
        "markersize": markersize,
        "linewidth": linewidth,
        "markevery": markevery["UniFastSvrg"],
    }

    return styles


# x-axis and y-axis labels
map_ylabel_dict = {
    "train_validation_metric": "f(x)-$f^\\star$",
    "val_acc": "test accuracy",
    "grad_norm": "$||\\nabla f(x)||$",
    "step_size": "Step size",
    "A": "A",
}

map_xlabel_dict = {
    "epoch": "Outer loops",
    "nb_backwards": "number of stochastic oracle calls ",
}


def plot_polyhedron(path_results, save_path):
    optims = [
        "AdaSVRG",
        "AdaVRAE",
        "AdaVRAG",
        "UniFastSvrg-AdaGrad",
        "UniSvrg-AdaGrad",
        "AccSVRG",
        "UniFastSgd-AdaGrad",
        "UniSgd-AdaGrad",
    ]
    order_labels = {
        "UniSgd (ours)": 1,
        "UniFastSgd (ours)": 2,
        "AdaSVRG": 3,
        "UniSvrg (ours)": 4,
        "AdaVRAG": 5,
        "AdaVRAE": 6,
        "FastSvrg": 7,
        "UniFastSvrg (ours)": 8,
    }

    marker_every = {
        "AdaSVRG": [3, 3, 2, 3],
        "AdaVRAE": [1000, 500, 200, 50],
        "AdaVRAG": [1000, 500, 200, 50],
        "UniFastSvrg-AdaGrad": [1000, 500, 200, 25],
        "UniSvrg-AdaGrad": [3, 3, 3, 3],
        "AccSVRG": [1000, 500, 200, 50],
        "UniFastSgd-AdaGrad": [1000, 500, 200, 75],
        "UniSgd-AdaGrad": [2000, 1000, 500, 200],
    }

    plot_config = {
        "nbrows": 1,
        "nbcols": 4,
        "linewidth": [2] * 4,
        "markersize": [8] * 4,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 4,
        "xscales": ["linear"] * 4,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.07),
        },
        "legend_index": 0,
        "figsize": (17, 4),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["q = 1", "q = 1.3", "q = 1.6", "q = 2"],
        "avg": "median",
        "filter_list": [optims] * 4,
        "ylim": [None] * 4,
        "xlim": [[-10000, 265000], [-5000, 150000], [-3000, 70000], [-1000, 19000]],
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_polyhedron,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/polyhedron-main.pdf"
    fig.savefig(save_fig, dpi=1600, bbox_inches="tight")


def plot_minibatch(path_results, save_path):
    optims = []
    markevery = {
        "UniFastSvrg-64": [20] * 4,
        "UniFastSvrg-512": [100] * 4,
        "UniFastSvrg-4096": [300] * 4,
        "UniSvrg-64": [3] * 4,
        "UniSvrg-512": [3] * 4,
        "UniSvrg-4096": [3] * 4,
        "UniFastSgd-64": [15] * 4,
        "UniFastSgd-512": [100] * 4,
        "UniFastSgd-4096": [700] * 4,
        "UniSgd-64": [30] * 4,
        "UniSgd-512": [300] * 4,
        "UniSgd-4096": [300] * 4,
    }

    order_labels = {"batchsize=64": 1, "batchsize=512": 2, "batchsize=4096": 3}

    plot_config = {
        "nbrows": 1,
        "nbcols": 1 * 4,
        "linewidth": [2] * 4,
        "markersize": [9] * 4,
        "markevery": markevery,
        "alpha": 0.2,
        "yscales": ["log"] * 4,
        "xscales": ["linear"] * 4,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.07),
        },
        "legend_index": [0, 1, 2, 3],
        "figsize": (22, 5),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 15,
            "fontweight": "bold",
        },
        "titles": ["UniSgd", "UniFastSgd", "UniSvrg", "UniFastSvrg"] * 4,
        "avg": "median",
        "filter_list": [optims] * 4,
        "ylim": [None] * 4,
        "xlim": [None, [-600, 15000], None, [-900, 20000]],
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4
    xlabels = ["nb_backwards"] * 4

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_minibatch,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/minibatch_polyhedron.pdf"
    fig.savefig(save_fig, dpi=1600, bbox_inches="tight")


def plot_logistic(path_results, save_path):
    optims1 = ["AdaSVRG", "UniSvrg-AdaGrad", "UniSgd-AdaGrad"]
    optims2 = ["AdaVRAE", "AdaVRAG", "UniFastSvrg-AdaGrad", "UniFastSgd-AdaGrad"]
    order_labels = {
        "UniSgd (ours)": 1,
        "UniFastSgd (ours)": 2,
        "AdaSVRG": 3,
        "UniSvrg (ours)": 4,
        "AdaVRAG": 5,
        "AdaVRAE": 6,
        "FastSvrg": 7,
        "UniFastSvrg (ours)": 8,
    }

    marker_every = {
        "AdaSVRG": [3, 2, 3, 3],
        "AdaVRAE": [0] * 4 + [25, 15, 500, 500],
        "AdaVRAG": [0] * 4 + [25, 15, 500, 500],
        "UniFastSvrg-AdaGrad": [0] * 4 + [25, 15, 500, 500],
        "UniSvrg-AdaGrad": [3, 3, 3, 3],
        "AccSVRG": [0] * 4 + [25, 15, 500, 500],
        "UniFastSgd-AdaGrad": [0] * 4 + [25, 25, 500, 500],
        "UniSgd-AdaGrad": [50, 5, 500, 500],
    }

    plot_config = {
        "nbrows": 2,
        "nbcols": 4,
        "linewidth": [2] * 8,
        "markersize": [7] * 8,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 8,
        "xscales": ["linear"] * 8,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.52, -0.05),
        },
        "legend_index": [0, 4],
        "figsize": (14, 6),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["mushrooms", "w8a", "leu", "colon-cancer", "", "", "", ""],
        "avg": "mean",
        "filter_list": [optims1] * 4 + [optims2] * 4,
        "ylim": [None] * 8,
        "xlim": [None, [-1000, 20000], [-3000, 75000], [-3000, 100000]] + [None] * 4,
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 8
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_polyhedron,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/logistic-regression.pdf"
    fig.savefig(save_fig, dpi=1600, bbox_inches="tight")


def plot_compare_polyhedron_non_acc(path_results, save_path):
    optims = ["UniSvrg-AdaGrad", "UniSgd-AdaGrad", "UniSgd", "UniSvrg"]
    order_labels = {
        "UniSgd-AdaGrad": 1,
        "UniSgd-Other": 2,
        "UniSvrg-AdaGrad": 3,
        "UniSvrg-Other": 4,
    }

    marker_every = {
        "UniSvrg-AdaGrad": [3, 3, 3, 3] * 2,
        "UniSvrg": [3, 3, 3, 3] * 2,
        "UniSgd-AdaGrad": [2000, 1000, 500, 200] * 2,
        "UniSgd": [2000, 1000, 500, 200] * 2,
        "UniFastSvrg-AdaGrad": [""] * 4 + [1000, 500, 200, 25],
        "UniFastSvrg": [""] * 4 + [1000, 500, 200, 75],
        "UniFastSgd-AdaGrad": [""] * 4 + [1000, 500, 200, 75],
        "UniFastSgd": [""] * 4 + [1000, 500, 200, 75],
    }

    plot_config = {
        "nbrows": 2,
        "nbcols": 4,
        "linewidth": [2] * 8,
        "markersize": [8] * 8,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 8,
        "xscales": ["linear"] * 8,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.05),
        },
        "legend_index": 0,
        "figsize": (17, 8),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["q = 1", "q = 1.3", "q = 1.6", "q = 2"] + [""] * 4,
        "avg": "median",
        "filter_list": [optims] * 8,
        "ylim": [None] * 8,
        "xlim": [[-10000, 265000], [-5000, 150000], [-3000, 70000], [-1000, 19000]] * 2,
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4 + ["step_size"] * 4
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_comp,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/compare-adagrad-main-non-acc-polyhedron.pdf"
    fig.savefig(
        save_fig,
        dpi=1600,
        bbox_inches="tight",
    )


def plot_compare_polyhedron_acc(path_results, save_path):
    optims = [
        "UniFastSvrg-AdaGrad",
        "UniFastSgd-AdaGrad",
        "UniFastSgd",
        "UniFastSvrg",
    ]
    order_labels = {
        "UniFastSgd-AdaGrad": 1,
        "UniFastSgd-Other": 2,
        "UniFastSvrg-AdaGrad": 3,
        "UniFastSvrg-Other": 4,
    }

    marker_every = {
        "UniSvrg-AdaGrad": [3, 3, 3, 3] * 2,
        "UniSvrg": [3, 3, 3, 3] * 2,
        "UniSgd-AdaGrad": [2000, 1000, 500, 200] * 2,
        "UniSgd": [2000, 1000, 500, 200] * 2,
        "UniFastSvrg-AdaGrad": [1000, 500, 200, 25] * 2,
        "UniFastSvrg": [1000, 500, 200, 75] * 2,
        "UniFastSgd-AdaGrad": [1000, 500, 200, 75] * 2,
        "UniFastSgd": [1000, 500, 200, 75] * 2,
    }

    plot_config = {
        "nbrows": 2,
        "nbcols": 4,
        "linewidth": [2] * 8,
        "markersize": [8] * 8,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 8,
        "xscales": ["linear"] * 8,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.05),
        },
        "legend_index": 0,
        "figsize": (17, 8),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["q = 1", "q = 1.3", "q = 1.6", "q = 2"] + [""] * 4,
        "avg": "median",
        "filter_list": [optims] * 8,
        "ylim": [None] * 8,
        "xlim": [[-10000, 265000], [-5000, 150000], [-3000, 70000], [-1000, 19000]] * 2,
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4 + ["step_size"] * 4
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_comp,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/compare-adagrad-main-acc-polyhedron.pdf"
    fig.savefig(
        save_fig,
        dpi=1600,
        bbox_inches="tight",
    )


def plot_compare_logistic_non_acc(path_results, save_path):
    optims = ["UniSvrg-AdaGrad", "UniSgd-AdaGrad", "UniSgd", "UniSvrg"]
    order_labels = {
        "UniSgd-AdaGrad": 1,
        "UniSgd-Other": 2,
        "UniSvrg-AdaGrad": 3,
        "UniSvrg-Other": 4,
    }

    marker_every = {
        "UniSvrg-AdaGrad": [3, 3, 3, 3] * 2,
        "UniSvrg": [3, 3, 3, 3] * 2,
        "UniSgd-AdaGrad": [50, 20, 500, 500] * 2,
        "UniSgd": [50, 20, 500, 500] * 2,
        "UniFastSvrg-AdaGrad": [""] * 4 + [25, 15, 500, 500],
        "UniFastSvrg": [""] * 4 + [25, 15, 500, 500],
        "UniFastSgd-AdaGrad": [""] * 4 + [25, 25, 500, 500],
        "UniFastSgd": [""] * 4 + [25, 25, 500, 500],
    }

    plot_config = {
        "nbrows": 2,
        "nbcols": 4,
        "linewidth": [2] * 8,
        "markersize": [8] * 8,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 8,
        "xscales": ["linear"] * 8,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.05),
        },
        "legend_index": 0,
        "figsize": (17, 8),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["q = 1", "q = 1.3", "q = 1.6", "q = 2"] + [""] * 4,
        "avg": "median",
        "filter_list": [optims] * 8,
        "ylim": [None] * 8,
        "xlim": [None, [-1000, 80000], [-3000, 75000], [-3000, 100000]] * 2,
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4 + ["step_size"] * 4
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_comp,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/compare-adagrad-main-non-acc-logis.pdf"
    fig.savefig(
        save_fig,
        dpi=1600,
        bbox_inches="tight",
    )


def plot_compare_logistic_acc(path_results, save_path):
    optims = [
        "UniFastSvrg-AdaGrad",
        "UniFastSgd-AdaGrad",
        "UniFastSgd",
        "UniFastSvrg",
    ]
    order_labels = {
        "UniFastSgd-AdaGrad": 1,
        "UniFastSgd-Other": 2,
        "UniFastSvrg-AdaGrad": 3,
        "UniFastSvrg-Other": 4,
    }

    marker_every = {
        "UniSvrg-AdaGrad": [3, 3, 3, 3] * 2,
        "UniSvrg": [3, 3, 3, 3] * 2,
        "UniSgd-AdaGrad": [50, 5, 500, 500] * 2,
        "UniSgd": [50, 5, 500, 500] * 2,
        "UniFastSvrg-AdaGrad": [25, 15, 500, 500] * 2,
        "UniFastSvrg": [25, 15, 500, 500] * 2,
        "UniFastSgd-AdaGrad": [25, 25, 500, 500] * 2,
        "UniFastSgd": [25, 25, 500, 500] * 2,
    }

    plot_config = {
        "nbrows": 2,
        "nbcols": 4,
        "linewidth": [2] * 8,
        "markersize": [8] * 8,
        "markevery": marker_every,
        "alpha": 0.2,
        "yscales": ["log"] * 8,
        "xscales": ["linear"] * 8,
        "legend_kwargs": {
            "loc": "outside lower center",
            "ncol": 12,
            "bbox_to_anchor": (0.5, -0.05),
        },
        "legend_index": 0,
        "figsize": (17, 8),
        "font_config": {
            "fontname": "Times New Roman",
            "size": 12,
            "fontweight": "bold",
        },
        "titles": ["q = 1", "q = 1.3", "q = 1.6", "q = 2"] + [""] * 4,
        "avg": "median",
        "filter_list": [optims] * 8,
        "ylim": [None] * 8,
        "xlim": [None, [-10000, 250000], [-8000, 150000], [-8000, 340000]] * 2,
        "order_labels": order_labels,
    }

    ylabels = ["train_validation_metric"] * 4 + ["step_size"] * 4
    xlabels = ["nb_backwards"] * 8

    fig, axs = plot_result(
        path_results,
        ylabels,
        xlabels,
        map_xlabel_dict,
        map_ylabel_dict,
        form_styles_comp,
        plot_config,
    )
    plt.tight_layout()
    save_fig = save_path + "/compare-adagrad-main-acc-logis.pdf"
    fig.savefig(
        save_fig,
        dpi=1600,
        bbox_inches="tight",
    )


def run_and_plot(args, savedir_results, savedir_figures, datadir, pw):
    # experiments in the main text
    experiments_main = [
        {
            "experiment_group": f"polyhedron_nd{i+1}",
            "savedir_base": savedir_results + f"/polyhedron/nd{i+1}",
            "datadir": datadir,
        }
        for i in range(4)
    ] + [
        {
            "experiment_group": f"minibatch_polyhedron_{i+1}",
            "savedir_base": savedir_results + f"/comp_minibatch_all/{i // 3 + 1}",
            "datadir": datadir,
        }
        for i in range(12)
    ]

    # experiments in the appendix
    datasets_logis = ["mushrooms", "w8a", "leu", "colon_cancer"]
    experiments_appendix = [
        {
            "experiment_group": f"logis_{dataset}",
            "savedir_base": savedir_results + f"/logis/{dataset}",
            "datadir": datadir,
        }
        for dataset in datasets_logis
    ]

    if pw == "main":
        experiments = experiments_main
    elif pw == "appendix":
        experiments = experiments_appendix
    else:
        experiments = experiments_main + experiments_appendix

    for experiment in experiments:
        args.exp_group_list = [experiment["experiment_group"]]
        args.datadir = experiment["datadir"]
        args.savedir_base = experiment["savedir_base"]
        hw.run_wizard(
            func=trainval,
            exp_groups=exp_configs.EXP_GROUPS,
            job_config=None,
            job_scheduler=None,
            python_binary_path=None,
            use_threads=True,
            args=args,
        )

    if pw == "main" or pw == "all":
        # plot figures in the main paper
        plot_polyhedron(
            path_results=[savedir_results + f"/polyhedron/nd{i+1}" for i in range(4)],
            save_path=savedir_figures,
        )
        plot_minibatch(
            path_results=[
                savedir_results + f"/comp_minibatch_all/{4-i}" for i in range(4)
            ],
            save_path=savedir_figures,
        )

    if pw == "appendix" or pw == "all":
        # plot figures in the appendix
        plot_logistic(
            path_results=[
                savedir_results + f"/logis/{dataset}" for dataset in datasets_logis
            ]
            * 2,
            save_path=savedir_figures,
        )
        plot_compare_polyhedron_non_acc(
            path_results=[savedir_results + f"/polyhedron/nd{i+1}" for i in range(4)]
            * 2,
            save_path=savedir_figures,
        )
        plot_compare_polyhedron_acc(
            path_results=[savedir_results + f"/polyhedron/nd{i+1}" for i in range(4)]
            * 2,
            save_path=savedir_figures,
        )
        plot_compare_logistic_non_acc(
            path_results=[
                savedir_results + f"/logis/{dataset}" for dataset in datasets_logis
            ]
            * 2,
            save_path=savedir_figures,
        )
        plot_compare_logistic_acc(
            path_results=[
                savedir_results + f"/logis/{dataset}" for dataset in datasets_logis
            ]
            * 2,
            save_path=savedir_figures,
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-pw",
        "--plot-which",
        default="all",
        choices=["main", "appendix", "all"],
        help="plot figures in the main paper, or in the appendix, or both",
    )
    parser.add_argument(
        "-sb",
        "--savedir-base",
        default="./results",
        help="base directory to save results and figures",
    )
    default_datadir = pooch.os_cache("universal_adagrad")
    parser.add_argument(
        "-d",
        "--datadir",
        default=default_datadir,
        help="directory to save datasets",
    )
    args, _ = parser.parse_known_args()
    args.cuda = 0
    pw = args.plot_which
    savedir_base = args.savedir_base
    savedir_results = savedir_base + "/results"
    savedir_figures = savedir_base + "/figures"
    datadir = args.datadir

    # make directories to save results and figures
    os.makedirs(savedir_base, exist_ok=True)
    os.makedirs(savedir_figures, exist_ok=True)
    os.makedirs(savedir_results, exist_ok=True)

    # run experiments and plot figures
    run_and_plot(args, savedir_results, savedir_figures, datadir, pw)
