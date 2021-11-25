from itertools import product
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, ax=None, savepath=None, cmap='viridis', include_values=True, scale_text=1, L_labels=None, y_labels=None, title=None):

    num_classes = max(cm.shape[0], cm.shape[1])
    text_size = 40*2*scale_text/num_classes
    ax_in = ax

    if ax is None: fig, ax = plt.subplots(figsize=(12, 12))

    if L_labels is not None and len(L_labels) != cm.shape[1]:
        raise ValueError
    
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text_[i, j] = ax.text(j, i, "{:0.2f}".format(cm[i, j]),
                                   ha="center", va="center",
                                   color=color, fontsize=text_size)

    ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       ylabel="y",
       xlabel="L")

    ax.tick_params(labelsize=text_size)
    ax.set_xlabel('L', fontsize=text_size+5)
    ax.set_ylabel('y', fontsize=text_size+5)

    ax.set_ylim((cm.shape[0] - 0.5, -0.5))
    
    if L_labels is not None:
        ax.set_xticklabels(L_labels)
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        plt.setp(ax.get_xticklabels(), rotation='horizontal')
        
    if y_labels is not None: ax.set_yticklabels(y_labels)
    
    if title is not None: 
        ax.set_title(title)
        ax.title.set_fontsize(text_size+5)

    if savepath is not None:
        plt.savefig(savepath)

    if ax_in is None: return fig, ax

        

# def plot_confusion_matrix(cm, savepath=None, cmap='viridis', include_values=True, L_labels=None):

#     num_classes = max(cm.shape[0], cm.shape[1])
#     text_size = 80*2/num_classes

#     if L_labels is not None and len(L_labels) != cm.shape[1]:
#         raise ValueError
    
#     fig, ax = plt.subplots(figsize=(12, 12))

#     im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     text_ = None

#     cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

#     if include_values:
#         text_ = np.empty_like(cm, dtype=object)

#         # print text with appropriate color depending on background
#         thresh = (cm.max() + cm.min()) / 2.0
#         for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
#             color = cmap_max if cm[i, j] < thresh else cmap_min
#             text_[i, j] = ax.text(j, i, "{:0.3f}".format(cm[i, j]),
#                                    ha="center", va="center",
#                                    color=color, fontsize=text_size)

#     ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        ylabel="y",
#        xlabel="L")

#     ax.tick_params(labelsize=20)
#     ax.set_xlabel('L', fontsize=30)
#     ax.set_ylabel('y', fontsize=30)
#     if L_labels is not None: ax.set_xticklabels(L_labels)

#     ax.xaxis.tick_top()
#     ax.xaxis.label_position = 'top'
#     ax.xaxis.labelpad = 40
#     ax.yaxis.labelpad = 20

#     ax.set_ylim((cm.shape[0] - 0.5, -0.5))
#     plt.setp(ax.get_xticklabels(), rotation='horizontal')

#     if savepath is not None:
#         plt.savefig(savepath)

#     return fig, ax
