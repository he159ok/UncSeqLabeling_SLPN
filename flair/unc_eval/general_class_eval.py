import os
import sys
from operator import itemgetter

import sklearn
import sklearn.metrics
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# import MyFunc as mf
from .MyFunc import cal_bnn_unc_scores, cal_lossaug_bnn_unc_scores, idk_un_metric, human_idk_un_metric, attention_masks

from .ECE import ece_score
# import MyFunc

def show_roc_auc(y_truth, y_pred, unc_list, cal_aupr = True, auroc_aupr_pos_id=None):  # 应该百分比
	# 使用的 y_pred的 one-hot形式
	# 应该是 prob形式
	# from sklearn.metrics import precision_recall_curve
	# from sklearn.metrics import average_precision_score
    y_truth = np.array(y_truth)
    y_pred_label = np.array(y_pred) # torch.argmax(y_pred_softmax, dim=1).numpy()
    if auroc_aupr_pos_id is None:
        y_binary = (y_pred_label == y_truth).astype(int)
        conf_array = 1.0 / (np.array(unc_list) + 1e-8)
    else:
        set_y_truth = set(y_truth)
        assert auroc_aupr_pos_id in set_y_truth
        y_binary = []
        for ele in y_truth:
            if ele == auroc_aupr_pos_id:
                y_binary.append(1)
            else:
                y_binary.append(0)
        conf_array = np.array(unc_list)

    # conf_array = 1.0 / (np.array(unc_list) + 1.0)
    fpr, tpr, thresholds_auroc = sklearn.metrics.roc_curve(y_binary, conf_array, pos_label=1)
    auroc_score = sklearn.metrics.auc(fpr, tpr)

    precision, recall, thresholds_aupr, aupr_score = None, None, None, None
    if cal_aupr:
        precision, recall, thresholds_aupr = sklearn.metrics.precision_recall_curve(y_binary, conf_array, pos_label=1)
        aupr_score = sklearn.metrics.auc(recall, precision)
    # plt.plot(precision, recall, color=color, label=str(toolsName)+'(AUPR = %0.3f)' % aupr, linestyle='--', LineWidth=3)#横纵坐标的取值，颜色样式等
    return auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score

def copy_show_roc_auc(y_truth, y_pred, unc_list, cal_aupr = True):  # 应该百分比

	y_truth = np.array(y_truth)
	y_pred_label = np.array(y_pred) # torch.argmax(y_pred_softmax, dim=1).numpy()
	y_binary = (y_pred_label == y_truth).astype(int) # wrong should be binary according to OOD label or not.



	conf_array = 1.0 / (np.array(unc_list)+1e-8)

	fpr, tpr, thresholds_auroc = sklearn.metrics.roc_curve(y_binary, conf_array, pos_label=1)
	auroc_score = sklearn.metrics.auc(fpr, tpr)

	precision, recall, thresholds_aupr, aupr_score = None, None, None, None

	if cal_aupr:
		precision, recall, thresholds_aupr = sklearn.metrics.precision_recall_curve(y_binary, conf_array, pos_label=1)
		aupr_score = sklearn.metrics.auc(recall, precision)

	return auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score



def dissonance_uncertainty(pred):
    alpha = np.array(pred)
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = alpha / S
    # belief = alpha
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / term_bj
            dis_un[k] += dis_ki
    return dis_un






def Bal(b_i, b_j):
    result = 1 - np.abs(b_i - b_j) / (b_i + b_j)
    return result


def vacuity_uncertainty(pred):
    # Vacuity uncertainty
    alpha = np.array(pred)
    class_num = alpha.shape[1]
    S = np.sum(alpha, axis=1, keepdims=True)
    un_vacuity = class_num / S
    return un_vacuity




def get_un_entropy(pred, mode='sfmx'):
    pred = np.array(pred)
    un = []
    dr_entroy, dr_entroy_class = cal_entropy(pred, mode)

    return dr_entroy

def softmax(pred):
    ex = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    return prob

def cal_entropy(pred, mode='sfmx'):
    class_num = pred.shape[1]
    if mode == 'sfmx':
        prob = softmax(pred)
    elif mode == 'mean':
        S = np.sum(pred, axis=1, keepdims=True)
        prob = pred / S
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def one_over_max(pred):
    one_over_max_unc = [
        1.0 / (max(ele) + 1e-8) for ele in pred
    ]
    return one_over_max_unc

def one_over_sum(pred):
    one_over_sum_unc = [
        1.0 / (sum(ele) + 1e-8) for ele in pred
    ]
    return one_over_sum_unc

def get_auroc_aupr(y_true, y_pred, enpy_unc, keyword, auroc_aupr_pos_id=None):
    writent_list = []
    auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(y_true,
                                                                                                           y_pred,
                                                                                                           enpy_unc,
                                                                                                           cal_aupr=True,
                                                                                                           auroc_aupr_pos_id=auroc_aupr_pos_id,
                                                                                                           )
    print(f"{keyword} auroc & aupr are {auroc_score}, {aupr_score}")
    writent_list.append("\n")
    writent_list.append(f"{keyword} auroc is {auroc_score}\n")
    writent_list.append(f"{keyword} aupr is  {aupr_score}\n")
    return auroc_score, aupr_score, writent_list

### below is the metrics for the dropout-based baselines
def entropy_dropout(pred):
    mean = []
    for p in pred:
        prob_i = softmax(p)
        mean.append(prob_i)
    mean = np.mean(mean, axis=0)
    class_num = mean.shape[1]
    prob = mean
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def aleatoric_dropout(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_softmax(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un

def entropy_softmax(pred):
    class_num = pred.shape[1]
    prob = softmax(pred)
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def get_un_dropout(pred): # used to get epistemic uncertainty in dropout
    un = []
    dr_entroy, dr_entroy_class = entropy_dropout(pred)
    dr_ale, dr_ale_clsss = aleatoric_dropout(pred)
    dr_eps_class = dr_entroy_class - dr_ale_clsss
    dr_eps = np.sum(dr_eps_class, axis=1, keepdims=True)
    un.append(dr_entroy)
    un.append(dr_ale)
    un.append(dr_eps)
    return un
