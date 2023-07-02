#!/usr/bin/env python
# encoding: utf-8

from ast import arg
from copy import copy, deepcopy
from dis import dis
import pdb
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from turtle import distance
import torch
import numpy as np
from functools import partial

from utils.common_utils import create_one_hot_labels, get_most_overlapped_element, check_numpy_to_torch
from chamfer_distance import ChamferDistance
from dataset_splitter import kl_divergence_distance, cal_probs2entropy

import torch
import torch.nn as nn
# use the CosineEmbeddingLoss as ConstrativeLoss implementations

min_var_est = 1e-8
sigma_list = [0.01, 0.1, 1, 10, 100]

def mmd_cal(label_s, feat_s, label_t, feat_t, args:dict, data_s=None, data_t=None, KPC=False, num_class=10):
    # Currently, lets use SOFT_MMD 
    sample_weights = None
    sample_weights_flag = args.get("GEO_WEIGHTS", None) or args.get("SEM_WEIGHTS", None)
    if data_s is not None and sample_weights_flag:
        sample_weights = cal_sample_weights(data_s, data_t, args, label_s=label_s, label_t=label_t, KPC=KPC, num_class=num_class)
        sample_weights = cal_sample_weights(data_s, data_t, args, label_s=label_s, label_t=label_t, num_class=num_class)
    if args["NAME"] == "SOFT_MMD":
        return soft_mmd(label_s, feat_s, label_t, feat_t, float(args["LABEL_SCALE"]), sample_weights=sample_weights, num_class=num_class)
    elif args["NAME"] == "HARD_MMD":
        return hard_mmd(label_s, feat_s, label_t, feat_t)
    elif args["NAME"]  == "MAX_HARD_MMD":
        return max_hard_mmd(label_s, feat_s, label_t, feat_t, num_class=num_class)
    elif args["NAME"] == "OFF":
        return mix_rbf_mmd2(feat_s, feat_t, sigma_list)
    else:
        raise RuntimeError("Not Supported MMD Method")


def cal_sample_weights(data_s, data_t, args, label_s=None, label_t=None, KPC=False, num_class=10):
    if args.get("GEO_WEIGHTS", None):
        sample_weights = geometric_weights(data_s, data_t, weighting=args["GEO_WEIGHTS"],KPC=KPC)
    elif args.get("ENTROPY_WEIGHTS", None):
        sample_weights = entropy_weights(data_s, data_t, weighting=args["ENTROPY_WEIGHTS"])
    elif args.get("SEM_WEIGHTS", None):
        sample_weights = prob_weights_soft(data_s, data_t, label_s, label_t, args["LABEL_WEIGHT"], args["SEM_WEIGHTS"], num_class=num_class)
    else:
        raise RuntimeError("Not suppprted weighting opperation")
    return sample_weights


def soft_mmd(label_s, feat_s, label_t, feat_t, label_weight, sample_weights=None, num_class=10):
    """
        First covert the scalar label to one-hot vector
        Concat the label vector (batch * 10) to the feat vector (batch * n) 4096 for geo-level and  256 for sem-level
    """
    label_s_one_hot = create_one_hot_labels(label_s, num_class=num_class).to(device='cuda')
    label_t_one_hot = create_one_hot_labels(label_t, num_class=num_class).to(device='cuda')
    feat_s_label = torch.cat((feat_s, label_s_one_hot * label_weight), dim=1)
    feat_t_label = torch.cat((feat_t, label_t_one_hot * label_weight), dim=1)

    return mix_rbf_mmd2(feat_s_label, feat_t_label, sigma_list, sample_weights=sample_weights)


def hard_mmd(label_s, feat_s, label_t, feat_t):
    """
        Direct use torch.eq to calculate the loss when same label occurs
    """
    same_class_index = torch.eq(label_s, label_t)
    selected_feat_node_s = feat_s[same_class_index]
    selected_feat_node_t = feat_t[same_class_index]

    return mix_rbf_mmd2(selected_feat_node_s, selected_feat_node_t, sigma_list)


def contrastive_loss_weighted(label_s, feat_s, label_t, feat_t, label_weight, CL_criterion, sample_weights=None, num_class=10):
    label_s_one_hot = create_one_hot_labels(label_s, num_class=num_class).to(device='cuda')
    label_t_one_hot = create_one_hot_labels(label_t, num_class=num_class).to(device='cuda')
    feat_s_label = torch.cat((feat_s, label_s_one_hot * label_weight), dim=1)
    feat_t_label = torch.cat((feat_t, label_t_one_hot * label_weight), dim=1)

    same_class_index = torch.eq(label_s, label_t)
    same_class_index = 2 * same_class_index - 1
    loss = CL_criterion(feat_s, feat_t, same_class_index)
    if sample_weights is not None:
        sample_weights = sample_weights.squeeze().to(device='cuda')
        loss = torch.mul(sample_weights, loss)
    loss = torch.mean(loss)

    return loss

def max_hard_mmd(label_s, feat_s, label_t, feat_t, num_class=10):
    """
        Try to have max class alignment and reorder the feature vector
    """
    ind_s, ind_t = get_most_overlapped_element(label_s.cpu(), label_t.cpu(), num_class=num_class)
    assert len(ind_s) == len(ind_t), "The feature shape mis-matched"
    selected_feat_node_s = feat_s[ind_s]
    selected_feat_node_t = feat_t[ind_t]
    return mix_rbf_mmd2(selected_feat_node_s, selected_feat_node_t, sigma_list)


def geometric_weights(pc_s, pc_t, metric="chamfer_distance", weighting="none", KPC=False):
    assert pc_s.shape[0] == pc_t.shape[0]
    criteria = None
    if metric != "chamfer_distance":
        raise RuntimeError("Currently Only Support CD distance") 
    if pc_s.shape[1] == 3:
        # currently, only support CD
    # batch * 3 * num_points -> batch * num_points * 3
        pc_1 = pc_s.transpose(1,2).squeeze()
        pc_2 = pc_t.transpose(1,2).squeeze()
    else:
        pc_1 = pc_s
        pc_2 = pc_t
    # to use the CD, the pc should be batch_size * num_points *3
    # return torch.tensor(batch_size)
    if  KPC:
        dist1, dist2, idx1, idx2 = ChamferDistance(pc_1, pc_2)
        distance = torch.mean(dist1, dim=1) +  torch.mean(dist2, dim=1)
    else:
        cd = ChamferDistance()
        criteria = partial(cd_distance, chamfer_dist=cd)
        distance = criteria(pc_1, pc_2)

    weights = distance2weights(distances=distance, method=weighting)
    return weights.reshape(1, -1)


def prob_weights_soft(pred_s, pred_t, label_s, label_t, label_weight, weighting="mean2one", num_class=10):
    assert label_weight < 1, "For Entropy, Label weight should be less than one"

    # need to convert logits to prob with softmax firstly, in order to use KL
    pred_s_ = torch.softmax(pred_s.detach().cpu(), dim=1)
    label_s_one_hot = create_one_hot_labels(label_s, num_class=num_class)
    pred_s_label = torch.cat((pred_s_.view(-1, num_class), label_s_one_hot * label_weight), dim=1)

    pred_t_ = torch.softmax(pred_t.detach().cpu(), dim=1)
    label_t_one_hot = create_one_hot_labels(label_t, num_class=num_class)
    pred_t_label = torch.cat((pred_t_.view(-1, num_class), label_t_one_hot * label_weight), dim=1)

    distance = kl_divergence_distance(normalized(pred_s_label), normalized(pred_t_label)).sum(1)
    weights = distance2weights(distances=distance, method=weighting)
    return weights.reshape(1, -1)


def normalized(vec):
    vec += min_var_est
    return vec / torch.sum(vec)

def entropy_weights(pred_s, pred_t, weighting="exp_inverse"):
    distance = entropy_dis(pred_s, pred_t)
    weights = distance2weights(distances=distance, method=weighting)
    return torch.tensor(np.array(weights).reshape(1, -1))


def entropy_dis(pred_s, pred_t):
    entropy_s = cal_probs2entropy(pred_s)
    entropy_t = cal_probs2entropy(pred_t)

    dis = kl_divergence_distance(entropy_s, entropy_t)
    return dis


def cd_distance(pc1, pc2, chamfer_dist, batch_loss=True):
    dist1, dist2, idx1, idx2 = chamfer_dist(pc1,pc2)
    if not batch_loss:
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
    else:
        loss = torch.mean(dist1, dim=1) +  torch.mean(dist2, dim=1)
    return loss


def distance2weights(distances, method="naive_inverse"):
    # Distances: List 
    # Return: 
    if method == "naive_inverse":
        nai_weights = [ 1 / (dis + min_var_est) for dis in distances]
        weights = [cls_weight / sum(nai_weights) for cls_weight in nai_weights]
    elif method == "exp_inverse":
        exp_cls = [np.exp(- dis ) for dis in distances]
        weights = [exp_cls_ / sum(exp_cls) for exp_cls_ in exp_cls]
    elif method == "hist":
        cls_weights = np.arange(1, 0, -0.1)
        weights = [0] * distances.shape[0]
        value_edges = np.histogram(distances, bins=cls_weights.shape[0])[1]
        for i in range(cls_weights.shape[0]):
            pos=np.where( (distances>= value_edges[i] ) & (distances<  value_edges[i+1]))
            weights[pos] = cls_weights[i]
    elif method == "none":
        weights = deepcopy(distances)
        # mmd would be much larger than naive one 

    elif method == "mean2one":
        # the mean to be one -> mean valued-pair is same to naive mmd
        scale_ = (1 /distances.mean()).type(torch.int)
        weights = distances * scale_
    return weights.reshape(-1, 1).squeeze()

# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    # pdb.set_trace()
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)
    # Z: 128 * 4096 + C (128 = 2*64)
    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t()) # 128 * 128
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True, sample_weights=None):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased, sample_weights=sample_weights)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, sample_weights=None):
    # sample_weights: batch_size * 1 -> scalar -> describe the weights that how much one x-y pair discrepancy and thus 
    # how much they should contribute to the final loss
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X) # scalar
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # batch_szie \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # batch_szie \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # batch_szie K_{XY}^T * e

    if sample_weights is not None:
        sample_weights = sample_weights.squeeze().to(device='cuda')
        assert sample_weights.shape[0] == K_XY_sums_0.shape[0]
        K_XY_sums_0 = torch.mul(sample_weights, K_XY_sums_0)

    Kt_XX_sum = Kt_XX_sums.sum()  # scalar e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal ** 2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY ** 2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    var_est = (
            2.0 / (m ** 2 * (m - 1.0) ** 2) * (
                2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
            - (4.0 * m - 6.0) / (m ** 3 * (m - 1.0) ** 3) * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 4.0 * (m - 2.0) / (m ** 3 * (m - 1.0) ** 2) * (
                        K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
            - 4.0 * (m - 3.0) / (m ** 3 * (m - 1.0) ** 2) * (K_XY_2_sum) - (8 * m - 12) / (
                        m ** 5 * (m - 1)) * K_XY_sum ** 2
            + 8.0 / (m ** 3 * (m - 1.0)) * (
                    1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                    - Kt_XX_sums.dot(K_XY_sums_1)
                    - Kt_YY_sums.dot(K_XY_sums_0))
    )
    return mmd2, var_est
