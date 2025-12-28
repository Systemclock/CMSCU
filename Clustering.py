'''
Multi-view clustering and evaluation
'''
import sys
import os

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster._supervised import check_clusterings
import scipy.io as sio
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from munkres import Munkres

from Metrics import *


def Clustering(x_list, y, w):
    # print('******** Clustering ********')
    n_clusters = np.size(np.unique(y))


    # for i in range(len(w)):
    #     print(w[i])
    # beta!=0
    for i in range(len(x_list)):
        x_list[i] = np.array(x_list[i], dtype=np.float32) * w[i]
        # print(np.array(x_list[i]).shape)
    x_final_concat = np.sum(x_list[:], axis=0)
    # sio.savemat('./save_data/scene/fuse_data.mat', {'data': x_final_concat})

    # X_con = x_final_concat
    # Y = y
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=500)
    # X_tsne = tsne.fit_transform(X_con)
    #
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # ¹éÒ»»¯
    # plt.figure(figsize=(8, 6))
    # colmap = plt.cm.get_cmap('viridis', 10)  # viridis tab10
    # plt.scatter(X_norm[:, 0], X_norm[:, 1], c=Y, cmap=colmap, s=10)
    #
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('./Figure/mnist_fuse.pdf', bbox_inches='tight')
    # print(x_final_concat.shape)
    # beta = 0
    # x_final_concat = np.concatenate(x_list[:], axis=1)  # ¼òµ¥Á¬½ÓÁËÃ¿¸öÊÔÍ¼ v*n*c -> n(vc)

    # print(np.shape(x_final_concat))
    # if dataset=mnist y=y
    # if np.min(y) == 1:
    #     y = y - 1
    y = y.detach().cpu().numpy()
    # y = y.cpu().numpy()
    # print(y[:20])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(x_final_concat)
    kmeas_assignments = kmeans.predict(x_final_concat)
    y_pred = get_y_preds(y, kmeas_assignments, n_clusters)
    # acc_score = metrics.accuracy_score(y, y_pred)
    acc_score = calculate_acc(y, y_pred)
    nmi_score = metrics.normalized_mutual_info_score(y, y_pred)
    # f = metrics.f1_score(y, y_pred, average='macro')
    a, b, f = b3_precision_recall_fscore(y, y_pred)

    return acc_score, nmi_score, f

def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    # u = linear_sum_assignment(w.max() - w)
    # ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)  # 混淆矩阵=(真实标签，预测标签)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    if verbose:
        # print('Confusion Matrix')
        # print(confusion_matrix)
        print('accuracy', accuracy, 'precision', precision, 'recall', recall, 'f_measure', f_score)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    if verbose:
        print('AMI', ami, 'NMI:', nmi, 'ARI:', ari)
    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    '''
    Using either a newly instantiated ClusterClass or a provided
    cluster_obj, generates cluster assignments based on input data

    x:              the points with which to perform clustering
    cluster_obj:    a pre-fitted instance of a clustering class
    ClusterClass:   a reference to the sklearn clustering class, necessary
                    if instantiating a new clustering class
    n_clusters:     number of clusters in the dataset, necessary
                    if instantiating new clustering class
    init_args:      any initialization arguments passed to ClusterClass

    returns:    a tuple containing the label assignments and the clustering object
    '''
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score