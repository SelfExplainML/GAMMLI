import os 
import numpy as np
import pandas as pd 
from contextlib import closing
from itertools import combinations
from itertools import product
import tensorflow as tf

import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from interpret.glassbox.ebm.utils import EBMUtils
from interpret.utils import autogen_schema
from interpret.glassbox.ebm.internal import NativeEBM
from interpret.glassbox.ebm.ebm import EBMPreprocessor


def get_interaction_list(tr_x, val_x, tr_y, val_y, pred_tr, pred_val, feature_list, feature_type_list, 
                 active_main_effect_index, user_feature_list, item_feature_list, task_type="Regression", interaction_restrict=None):

    if task_type == "Regression":
        num_classes_ = -1
        model_type = "regression"
    elif task_type == "Classification":
        num_classes_ = 2
        model_type = "classification"
        pred_tr = np.minimum(np.maximum(pred_tr, 0.0000001), 0.9999999)
        pred_val = np.minimum(np.maximum(pred_val, 0.0000001), 0.9999999)
        pred_tr = np.log(pred_tr / (1 - pred_tr)) 
        pred_val = np.log(pred_val / (1 - pred_val)) 
        
    train_num = tr_x.shape[0]
    x = np.vstack([tr_x, val_x])
    schema_ = autogen_schema(pd.DataFrame(x), feature_names=feature_list, feature_types=feature_type_list)
    preprocessor_ = EBMPreprocessor(schema=schema_)
    preprocessor_.fit(x)
    xt = preprocessor_.transform(x)
    
    tr_x, val_x = xt[:train_num, :], xt[train_num:, :]
    attributes_ = EBMUtils.gen_attributes(preprocessor_.col_types_, preprocessor_.col_n_bins_)
    main_attr_sets = EBMUtils.gen_attribute_sets([[item] for item in range(len(attributes_))])

    with closing(
        NativeEBM(
            attributes_,
            main_attr_sets,
            tr_x,
            tr_y,
            val_x,
            val_y,
            num_inner_bags=0,
            num_classification_states=num_classes_,
            model_type=model_type,
            training_scores=pred_tr,
            validation_scores=pred_val,
        )
    ) as native_ebm:

        interaction_scores = []
        interaction_scores = []
        if interaction_restrict=='inter':
            interaction_indices = [item for item in combinations(user_feature_list,2)] + [item for item in combinations(item_feature_list,2)]
            
        elif interaction_restrict=='intra':
            interaction_indices = [item for item in product(user_feature_list, item_feature_list)]
        else:
            interaction_indices = [item for item in combinations(range(len(preprocessor_.col_types_)), 2)]
            
        for pair in interaction_indices:
            if (pair[0] in active_main_effect_index) or (pair[1] in active_main_effect_index):
                score = native_ebm.fast_interaction_score(pair)
                interaction_scores.append((pair, score))

        ranked_scores = list(
            sorted(interaction_scores, key=lambda item: item[1], reverse=True)
        )

    interaction_list = [ranked_scores[i][0] for i in range(len(ranked_scores))]
    return interaction_list

def plot_regularization(data_dict_logs, log_scale=True, save_eps=False, save_png=False, folder="./results/", name="trajectory_plot"):

    main_loss = data_dict_logs["main_effect_val_loss"]
    inter_loss = data_dict_logs["interaction_val_loss"]
    active_main_effect_index = data_dict_logs["active_main_effect_index"]
    active_interaction_index = data_dict_logs["active_interaction_index"]
    
    fig = plt.figure(figsize=(14, 4))
    if len(main_loss) > 0:
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(np.arange(0, len(main_loss), 1), main_loss)
        ax1.axvline(np.argmin(main_loss), linestyle="dotted", color="red")
        ax1.axvline(len(active_main_effect_index), linestyle="dotted", color="red")
        ax1.plot(np.argmin(main_loss), np.min(main_loss), "*", markersize=12, color="red")
        ax1.plot(len(active_main_effect_index), main_loss[len(active_main_effect_index)], "o", markersize=8, color="red")
        ax1.set_xlabel("Number of Main Effects", fontsize=12)
        ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        ax1.set_xlim(-0.5, len(main_loss) - 0.5)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(main_loss)), np.log10(np.nanmax(main_loss)), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Training Loss (Log Scale)", fontsize=12)
        else:
            ax1.set_yticks((np.linspace(np.nanmin(main_loss), np.nanmax(main_loss), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Training Loss", fontsize=12)

    if len(inter_loss) > 0:
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(0, len(inter_loss), 1), inter_loss)
        ax2.axvline(np.argmin(inter_loss), linestyle="dotted", color="red")
        ax2.axvline(len(active_interaction_index), linestyle="dotted", color="red")
        ax2.plot(np.argmin(inter_loss), np.min(inter_loss), "*", markersize=12, color="red")
        ax2.plot(len(active_interaction_index), inter_loss[len(active_interaction_index)], "o", markersize=8, color="red")
        ax2.set_xlabel("Number of Interactions", fontsize=12)
        ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        ax2.set_xlim(-0.5, len(inter_loss) - 0.5)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax2.set_yscale("log")
            ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(inter_loss)), np.log10(np.nanmax(inter_loss)), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        else:
            ax2.set_yticks((np.linspace(np.nanmin(inter_loss), np.nanmax(inter_loss), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss", fontsize=12)
    plt.show()

    save_path = folder + name
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_eps:
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def plot_trajectory(data_dict_logs, log_scale=True, save_eps=False, save_png=False, folder="./results/", name="trajectory_plot"):

    t1, t2, t3, t4 = [data_dict_logs["err_train_main_effect_training"],
                data_dict_logs["err_train_interaction_training"], data_dict_logs["err_train_tuning"], data_dict_logs["err_train_mf"]]
    v1, v2, v3, v4 = [data_dict_logs["err_val_main_effect_training"],
                data_dict_logs["err_val_interaction_training"], data_dict_logs["err_val_tuning"], data_dict_logs["err_val_mf"]]
    
    offset1x = (0.25 - len(t1) / len(t1 + t2 + t3 + t4 )) * 300
    offset2x = (0.45 - len(t1 + t2) / len(t1 + t2 + t3 + t4 )) * 300
    offset3x = (0.65 - len(t1 + t2 + t3) / len(t1 + t2 + t3 + t4)) * 300
    
    fig = plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(t1) + 1, 1), t1, color="r")
    ax1.plot(np.arange(len(t1) + 1, len(t1 + t2 + t3 ) + 1, 1), t2 + t3, color="b")
    ax1.plot(np.arange(len(t1 + t2 + t3 ),len(t1 + t2 + t3 + t4 ),1), t4 ,color='g')
    if log_scale:
        ax1.set_yscale("log")
        ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(t1 + t2 + t3 + t4 )), np.log10(np.nanmax(t1 + t2 + t3 + t4)), 6)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss (Log Scale)", fontsize=12)
        hp1 = ((np.log10(t1[-1]) - np.log10(np.min(t1 + t2 + t3 + t4))) 
            / (np.log10(np.max(t1 + t2 + t3 + t4 )) - np.log10(np.min(t1 + t2 + t3 + t4 ))))
        hp2 = ((np.log10((t1 + t2)[-1]) - np.log10(np.min(t1 + t2 + t3 + t4 ))) 
            / (np.log10(np.max(t1 + t2 + t3 + t4 )) - np.log10(np.min(t1 + t2 + t3 + t4 ))))
        hp3 = ((np.log10((t1 + t2 + t3)[-1]) - np.log10(np.min(t1 + t2 + t3 + t4 ))) 
            / (np.log10(np.max(t1 + t2 + t3 + t4 )) - np.log10(np.min(t1 + t2 + t3 + t4 ))))

    else:
        ax1.set_yticks((np.linspace(np.nanmin(t1 + t2), np.nanmax(t1 + t2), 6)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss", fontsize=12)
        hp1 = (t1[-1] - np.min(t1 + t2 + t3 + t4 )) / (np.max(t1 + t2 + t3 + t4 ) - np.min(t1 + t2 + t3 + t4))
        hp2 = ((t1 + t2)[-1] - np.min(t1 + t2 + t3 + t4 )) / (np.max(t1 + t2 + t3 + t4 ) - np.min(t1 + t2 + t3 + t4 ))
        hp3 = ((t1 + t2 + t3)[-1] - np.min(t1 + t2 + t3 + t4 )) / (np.max(t1 + t2 + t3 + t4 ) - np.min(t1 + t2 + t3 + t4 ))
        
    offset1y = 65 if hp1 < 0.6 else -65
    offset2y = 65 if hp2 < 0.6 else -65
    offset3y = 65 if hp3 < 0.6 else -65
    
    """
    if len(t2) > 0:
        ax1.annotate("Add \n Manifest \n Interactions", ((len(t1) + 1), t1[-1]), xycoords="data",
                 xytext=(offset1x, offset1y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                 horizontalalignment="center", verticalalignment="top")
    if len(t3) > 0:
        ax1.annotate("Prune ", ((len(t1 + t2) + 1), (t1 + t2)[-1]), xycoords="data",
                 xytext=(offset2x, offset2y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                 horizontalalignment="center", verticalalignment="top")
    if len(t4) > 0:
        ax1.annotate("Add \n Latent \n Interactions", ((len(t1 + t2 + t3) + 1), (t1 + t2 + t3)[-1]), xycoords="data",
                xytext=(offset3x, offset3y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                horizontalalignment="center", verticalalignment="top")
    """

    ax1.legend(["Stage 1: Training Main Effects", "Stage 2: Training Manifest Interactions", "Stage 3: Training Latent Interactions"])

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.arange(1, len(v1) + 1, 1), v1 , color="r")
    ax2.plot(np.arange(len(v1) + 1, len(v1 + v2 + v3 ) + 1, 1), v2 + v3, color="b")
    ax2.plot(np.arange(len(v1 + v2 + v3 ) + 1, len(v1 + v2 + v3 + v4 ) + 1, 1), v4, color="g")
    if log_scale:
        ax2.set_yscale("log")
        ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(v1 + v2 + v3 + v4 )), np.log10(np.nanmax(v1 + v2 + v3 + v4 )), 6)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        hp1 = ((np.log10(v1[-1]) - np.log10(np.min(v1 + v2 + v3 + v4 )))
            / (np.log10(np.max(v1 + v2 + v3 + v4 )) - np.log10(np.min(v1 + v2 + v3 + v4 ))))
        hp2 = ((np.log10((v1 + v2)[-1]) - np.log10(np.min(v1 + v2 + v3 + v4 ))) 
            / (np.log10(np.max(v1 + v2 + v3 + v4 )) - np.log10(np.min(v1 + v2 + v3 + v4 ))))
        hp3 = ((np.log10((v1 + v2 + v3)[-1]) - np.log10(np.min(v1 + v2 + v3 + v4 ))) 
            / (np.log10(np.max(v1 + v2 + v3 + v4 )) - np.log10(np.min(v1 + v2 + v3 + v4 ))))
    else:
        ax2.set_yticks((np.linspace(np.nanmin(v1 + v2 + v3 + v4 + v5), np.nanmax(v1 + v2 + v3 + v4 + v5), 6)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss", fontsize=12)
        hp1 = (v1[-1] - np.min(v1 + v2 + v3 + v4)) / (np.max(v1 + v2 + v3 + v4) - np.min(v1 + v2 + v3 + v4))
        hp2 = ((v1 + v2)[-1] - np.min(v1 + v2 + v3 + v4)) / (np.max(v1 + v2 + v3 + v4) - np.min(v1 + v2 + v3 + v4))
        hp3 = ((v1 + v2 + v3)[-1] - np.min(v1 + v2 + v3 + v4)) / (np.max(v1 + v2 + v3 + v4) - np.min(v1 + v2 + v3 + v4))
        
    offset1y = 65 if hp1 < 0.6 else -65
    offset2y = 65 if hp2 < 0.6 else -65
    offset3y = 65 if hp3 < 0.6 else -65
    """
    if len(v2) > 0:
        ax2.annotate("Add \n Manifest \n Interactions", ((len(v1) + 1), v1[-1]), xycoords="data",
                xytext=(offset1x, offset1y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                horizontalalignment="center", verticalalignment="top")
    if len(v3) > 0:
        ax2.annotate("Prune ", ((len(v1 + v2) + 1), (v1 + v2)[-1]), xycoords="data",
                xytext=(offset2x, offset2y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                horizontalalignment="center", verticalalignment="top")
    if len(v4) > 0:
        ax2.annotate("Add \n Latent \n Interactions", ((len(v1 + v2 + v3) + 1), (v1 + v2 + v3)[-1]), xycoords="data",
                xytext=(offset3x, offset3y), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                horizontalalignment="center", verticalalignment="top")
    """

    ax2.legend(["Stage 1: Training Main Effects", "Stage 2: Training Manifest Interactions", "Stage 3: Training Latent Interactions"])
    plt.show()
    
    save_path = folder + name
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_eps:
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def feature_importance_visualize(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):

    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)
            
    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(7, 0.4 + 0.6 * max_ids))
        ax = plt.axes()
        rects = ax.barh(np.arange(len(all_ir)), [ir for ir,_ in sorted(zip(all_ir, all_names))])
        ax.set_yticks(np.arange(len(all_ir)))
        ax.set_yticklabels([name for _,name in sorted(zip(all_ir, all_names))])
        for rect in rects:
            _, height = rect.get_xy()
            ax.text(rect.get_x() + rect.get_width() + 0.005, height + 0.3,
                    "%0.3f" % (rect.get_width()))
        plt.ylabel("Feature Name", fontsize=12)
        plt.xlim(0, np.max(all_ir) + 0.05)
        plt.ylim(-1, len(all_names))
        plt.title("Feature Importance")

        save_path = folder + name
        if (max_ids > 0) & save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if (max_ids > 0) & save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def global_visualize_density(data_dict_global, main_effect_num=10**5, interaction_num=10**5, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./results/", name="demo"):

    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum()>0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    idx = 0
    fig = plt.figure(figsize=(6 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
    #fig.suptitle('Main Effects & Manifest Interactions',fontsize=25)
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0]) 
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1]) 
            xint = ((np.array(data_dict_global[feature_name]["density"]["names"][1:]) 
                            + np.array(data_dict_global[feature_name]["density"]["names"][:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax2.bar(xint, data_dict_global[feature_name]["density"]["scores"], width=xint[1] - xint[0])
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)

        elif data_dict_global[feature_name]["type"] == "categorical":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx],
                                        wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            ax2.bar(np.arange(len(data_dict_global[feature_name]["density"]["names"])),
                    data_dict_global[feature_name]["density"]["scores"])
            ax1.get_shared_x_axes().join(ax1, ax2)

            ax2.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax2.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)
            
        idx = idx + 1
        if len(str(ax2.get_xticks())) > 60:
            ax2.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=14)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        feature_name1 = feature_name.split(" vs. ")[0]
        feature_name2 = feature_name.split(" vs. ")[1]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[idx],
                                wspace=0.1, hspace=0.1, height_ratios=[6, 1], width_ratios=[0.6, 3, 0.15, 0.2])
        ax_main = plt.Subplot(fig, inner[1])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)
        ax_main.set_xticklabels([])
        ax_main.set_yticklabels([])
        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=14)
        fig.add_subplot(ax_main)

        ax_bottom = plt.Subplot(fig, inner[5])
        if data_dict_global[feature_name]["xtype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name1]["density"]["names"]))
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"])
            ax_bottom.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_bottom.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name1]["density"]["names"][1:]) 
                  + np.array(data_dict_global[feature_name1]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"], width=xint[1] - xint[0])
        ax_bottom.set_yticklabels([])
        ax_bottom.set_xlim([axis_extent[0], axis_extent[1]])
        ax_bottom.get_shared_x_axes().join(ax_bottom, ax_main)
        fig.add_subplot(ax_bottom)
        if len(str(ax_bottom.get_xticks())) > 60:
            ax_bottom.xaxis.set_tick_params(rotation=20)

        ax_left = plt.Subplot(fig, inner[0])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name2]["density"]["names"]))
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"])
            ax_left.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_left.set_yticklabels(data_dict_global[feature_name]["input2_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name2]["density"]["names"][1:]) 
                  + np.array(data_dict_global[feature_name2]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"], height=xint[1] - xint[0])
        ax_left.set_xticklabels([])
        ax_left.set_ylim([axis_extent[2], axis_extent[3]])
        ax_left.get_shared_y_axes().join(ax_left, ax_main)
        fig.add_subplot(ax_left)

        ax_colorbar = plt.Subplot(fig, inner[2])
        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"]) 
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        fig.colorbar(interact_plot, cax=ax_colorbar, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)
        fig.add_subplot(ax_colorbar)
        idx = idx + 1

    save_path = folder + name
    if (max_ids > 0) & save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if (max_ids > 0) & save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def global_visualize_wo_density(data_dict_global, main_effect_num=10**5, interaction_num=10**5, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./results/", name="demo"):

    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum()>0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    idx = 0
    fig = plt.figure(figsize=(5.2 * cols_per_row, 4 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids/cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            ax1 = plt.Subplot(fig, outer[idx]) 
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            fig.add_subplot(ax1)
            if len(str(ax1.get_xticks())) > 80:
                ax1.xaxis.set_tick_params(rotation=20)

        elif data_dict_global[feature_name]["type"] == "categorical":

            ax1 = plt.Subplot(fig, outer[idx]) 
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            ax1.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax1.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            fig.add_subplot(ax1)
        
        idx = idx + 1
        if len(str(ax1.get_xticks())) > 60:
            ax1.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        feature_name1 = feature_name.split(" vs. ")[0]
        feature_name2 = feature_name.split(" vs. ")[1]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        ax_main = plt.Subplot(fig, outer[idx])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)

        if data_dict_global[feature_name]["xtype"] == "categorical":
            ax_main.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_main.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            ax_main.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_main.set_yticklabels(data_dict_global[feature_name]["input2_labels"])

        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"]) 
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        fig.colorbar(interact_plot, ax=ax_main, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)

        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)
        fig.add_subplot(ax_main)
        
        idx = idx + 1
        if len(str(ax_main.get_xticks())) > 60:
                ax_main.xaxis.set_tick_params(rotation=20)
        

    save_path = folder + name
    if (max_ids > 0) & save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if (max_ids > 0) & save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def local_visualize(data_dict_local, folder="./results/", name="demo", save_png=False, save_eps=False , task_type='Regression'):

    left_x,left_y=0.1,0.1
    width,height=1,1.5
    #left_xh=left_x+width+0.2
    #left_xhh=left_xh+0.5

    scatter_area=[left_x,left_y,width,height]
    #hist_y=[left_xh,left_y,0.2,height]
    #hist_yy = [left_xhh,left_y,0.2,height]

    plt.figure(figsize=(6, round((len(data_dict_local["active_indice"]) + 1) * 0.45)))

    area_scatter=plt.axes(scatter_area)
    #area_histy=plt.axes(hist_y)
    #area_histyy=plt.axes(hist_yy)

    
    final_pre = data_dict_local["predicted"]+data_dict_local["scores"][0]
    if task_type== 'Classification':
        final_pre = data_dict_local["initial_predict"][0]+data_dict_local["scores"][0]
        final_pre = tf.sigmoid(final_pre).numpy()[0]
    if "actual" in data_dict_local.keys():
        area_scatter.set_title("Final Predicted: %0.4f | Actual: %0.4f" % (final_pre, data_dict_local["actual"]))
        #area_histyy.set_title("Final Predicted: %0.4f | Actual: %0.4f" % (final_pre, data_dict_local["actual"]))
    else:
        area_scatter.set_title("Fianl Predicted: %0.4f"% (final_pre))
        #area_histyy.set_title("Final Predicted: %0.4f"% (final_pre))
    
    area_scatter.barh(data_dict_local["effect_names"][data_dict_local["active_indice"]][::-1].tolist(), data_dict_local["scores"][data_dict_local["active_indice"]][::-1], )
    area_scatter.set_yticks(np.arange(len(data_dict_local["active_indice"])))
    
    #area_histy.set_title('importance')
    #if task_type == 'Classification':        
    #    area_histy.pie([abs(data_dict_local['initial_predict'][0][0])*100,abs(data_dict_local['scores'][0])*100],labels=['explicit','implicit'],shadow=True,explode=[0,1],autopct="%0.2f%%")
    #    area_histyy.bar(['explicit','implicit'],[data_dict_local['initial_predict'][0][0],data_dict_local['scores'][0]])
    #else:
    #    area_histy.pie([abs(data_dict_local['predicted'][0][0])*100,abs(data_dict_local['scores'][0])*100],labels=['explicit','implicit'],shadow=True,explode=[0,1],autopct="%0.2f%%")
    #    area_histyy.bar(['explicit','implicit'],[data_dict_local['predicted'][0][0],data_dict_local['scores'][0]])
    
    if save_eps:
        plt.savefig("local.eps", bbox_inches="tight", dpi=100)

