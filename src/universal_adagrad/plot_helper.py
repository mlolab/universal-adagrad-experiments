import io
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch


def sort_by_indexes(lst, indexes, reverse=False):
    return [
        val
        for (_, val) in sorted(zip(indexes, lst), key=lambda x: x[0], reverse=reverse)
    ]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def form_df(path_result, xlabel, ylabel, avg="mean", filter_list=[]):
    dfx = pd.DataFrame()
    dfy = pd.DataFrame()
    opt_names = []
    run_times = 0
    for exp in os.listdir(path_result):
        if exp != ".DS_Store":
            path_exp_dict = os.path.join(path_result, exp, "exp_dict.json")
            exp_dict = pd.read_json(path_exp_dict)
            opt_name = exp_dict["opt"]["name"]
            if (len(filter_list) > 0 and opt_name in filter_list) or (
                len(filter_list) == 0
            ):
                score = os.path.join(path_result, exp, "score_list.pkl")
                with open(score, "rb") as pickle_file:
                    content = CPU_Unpickler(pickle_file).load()
                df_score = pd.DataFrame(content)
                opt_names.append(opt_name)
                runs = exp_dict["runs"]["name"]
                run_times = max(run_times, runs)
                dfy = pd.concat(
                    [dfy, pd.DataFrame({opt_name + str(runs): df_score[ylabel]})],
                    axis=1,
                )
                dfx = pd.concat(
                    [dfx, pd.DataFrame({opt_name + str(runs): df_score[xlabel]})],
                    axis=1,
                )

    if avg is not None:
        min_max = {}
        dfx_mean = pd.DataFrame()
        dfy_mean = pd.DataFrame()
        for optim in set(opt_names):
            dfx_filter = dfx.filter(
                items=[optim + str(i) for i in range(run_times + 1)]
            )
            dfy_filter = dfy.filter(
                items=[optim + str(i) for i in range(run_times + 1)]
            )
            min_max[optim] = pd.concat(
                [dfy_filter.min(axis=1), dfy_filter.max(axis=1)], axis=1
            )

            if avg == "mean":
                dfx_mean = pd.concat(
                    [dfx_mean, pd.DataFrame({optim: dfx_filter.mean(axis=1)})], axis=1
                )
                dfy_mean = pd.concat(
                    [dfy_mean, pd.DataFrame({optim: dfy_filter.mean(axis=1)})], axis=1
                )
            elif avg == "median":
                dfx_mean = pd.concat(
                    [dfx_mean, pd.DataFrame({optim: dfx_filter.median(axis=1)})], axis=1
                )
                dfy_mean = pd.concat(
                    [dfy_mean, pd.DataFrame({optim: dfy_filter.median(axis=1)})], axis=1
                )

        return dfx_mean, dfy_mean, min_max

    else:
        return dfx, dfy


def plot_result(
    path_results,
    ylabels,
    xlabels,
    map_xlabel_dict,
    map_ylabel_dict,
    form_styles,
    plot_config,
):
    styles = form_styles(
        plot_config["linewidth"], plot_config["markevery"], plot_config["markersize"]
    )

    nbrows = plot_config["nbrows"]
    nbcols = plot_config["nbcols"]

    fig, axs = plt.subplots(nbrows, nbcols, figsize=plot_config["figsize"])
    axslist = []
    if nbrows == 1:
        if nbcols == 1:
            axslist.append(axs)
        else:
            for c in range(nbcols):
                axslist.append(axs[c])
    else:
        if nbcols == 1:
            for c in range(nbrows):
                axslist.append(axs[c])
        else:
            for r in range(nbrows):
                for c in range(nbcols):
                    axslist.append(axs[r, c])

    for index in range(nbrows * nbcols):
        dfx, dfy, min_max = form_df(
            path_results[index],
            xlabels[index],
            ylabels[index],
            avg=plot_config["avg"],
            filter_list=plot_config["filter_list"][index],
        )
        for optim in list(dfx.columns):
            axslist[index].fill_between(
                dfx[optim],
                min_max[optim][0],
                min_max[optim][1],
                alpha=plot_config["alpha"],
                fc=styles[optim]["color"],
            )
            axslist[index].plot(
                dfx[optim],
                dfy[optim],
                marker=styles[optim]["marker"],
                markersize=styles[optim]["markersize"][index],
                markevery=styles[optim]["markevery"][index],
                linewidth=styles[optim]["linewidth"][index],
                label=styles[optim]["label"],
                color=styles[optim]["color"],
            )
            axslist[index].set_xlabel(map_xlabel_dict[xlabels[index]])
            axslist[index].set_ylabel(map_ylabel_dict[ylabels[index]])
            axslist[index].set_title(
                plot_config["titles"][index], **plot_config["font_config"]
            )
            axslist[index].set_yscale(plot_config["yscales"][index])
            axslist[index].set_xscale(plot_config["xscales"][index])
            if plot_config["ylim"][index] is not None:
                axslist[index].set_ylim(plot_config["ylim"][index])
            if plot_config["xlim"][index] is not None:
                axslist[index].set_xlim(plot_config["xlim"][index])
            axslist[index].grid(True)

    if isinstance(plot_config["legend_index"], int):
        plot_config["legend_index"] = [plot_config["legend_index"]]
    handles = []
    labels = []
    for index in plot_config["legend_index"]:
        handle, label = axslist[index].get_legend_handles_labels()
        handles += handle
        labels += label
    if plot_config["order_labels"] is not None:
        order_list = []
        for label in labels:
            order_list.append(plot_config["order_labels"][label])
        handles = sort_by_indexes(handles, order_list)
        labels = sort_by_indexes(labels, order_list)

    fig.legend(handles, labels, **plot_config["legend_kwargs"])

    return fig, axs
