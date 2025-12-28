import random
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from munkres import Munkres
from sklearn import manifold
import matplotlib.pyplot as plt
def random_index(n_all, n_train, seed):
    '''
    generated random index
    '''
    random.seed(seed)
    random_index = random.sample(range(n_all), n_all)
    train_index = random_index[0:n_train]
    test_index = random_index[n_train:n_all]

    train_index = np.array(train_index)
    test_index = np.array(test_index)
    return train_index, test_index

def normalize(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x

def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """

    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

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
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None) 
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments)!=0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def _get_affinity_matrix(X: torch.Tensor, K) -> torch.Tensor:
    # calculate adaptive affinity matrix
    eps = 2.2204e-16
    # eps = 1e-7
    n_neighbors = K  # neighbors num: 2 3 5
    X = X.cpu().detach().numpy()
    n, dim = X.shape
    D = EuDist2(X, X, squared=True)
    NN_full = np.argsort(D, axis=1)
    W = np.zeros((n,n))
    for i in range(n):
        id = NN_full[i, 1:(n_neighbors + 2)]
        di = D[i, id]
        W[i, id] = (di[-1] - di) / (n_neighbors * di[-1] - sum(di[:-1]) + eps)
    W = torch.tensor((W+W.T)/2)
    # W = torch.tensor(norm_W(W), dtype=torch.float32)

    # full connected
    # dists = torch.cdist(X, X)
    # W = torch.exp(-1 * 21.5 * (dists ** 2))
    # W.fill_diagonal_(0)
    # W = 0.5 * (W + W.T)

    # m = X.shape[0]
    # if not torch.is_tensor(X):
    #     X = torch.tensor(X, dtype=float)
    # n_ngb = 20
    # dists = torch.cdist(X, X)  # .toarray()
    # nn = torch.topk(-dists, n_ngb, sorted=True)
    # vals = nn[0]
    # scales = -vals[:, - 1]
    # const = X.shape[0] // 2
    # scales = torch.topk(scales, const)[0]
    # scale = scales[const - 1]
    # vals = vals / (2 * scale)
    # aff = torch.exp(vals)
    #
    # idx = nn[1]
    #
    # W = torch.zeros(m, m)
    # W = W.float()
    # aff = aff.float()
    # W[np.arange(m)[:, None], idx] = aff
    #
    # W = 0.5 * (W + W.T)
    # W.fill_diagonal_(0)


    # KNN
    # m = X.shape[0]
    # if not torch.is_tensor(X):
    #     X = torch.tensor(X, dtype=float)
    # n_ngb = 20
    # dists = torch.cdist(X, X)  # .toarray()
    # nn = torch.topk(-dists, n_ngb, sorted=True)
    # vals = nn[0]
    # scales = -vals[:, - 1]
    # const = X.shape[0] // 2
    # scales = torch.topk(scales, const)[0]
    # scale = scales[const - 1]
    # vals = vals / (2 * scale)
    # aff = torch.exp(vals)
    #
    # idx = nn[1]
    #
    # W = torch.zeros(m, m)
    # W = W.float()
    # aff = aff.float()
    # W[np.arange(m)[:, None], idx] = aff
    #
    # W = 0.5 * (W + W.T)
    # W.fill_diagonal_(0)

    return W

def norm_W(A):
    A = A.cpu().detach().numpy()
    d = np.sum(A, 1)
    d[d == 0] = 1e-6
    d_inv = 1 / np.sqrt(d)
    tmp = A * np.outer(d_inv, d_inv)
    A2 = np.maximum(tmp, tmp.T)
    return A2

def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids

def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale

def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales

        W = torch.exp(
            -1 * 21.5 * torch.pow(D, 2).to(device)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W

def make_batches(size, batch_size):
    '''
    generates a list of (start_idx, end_idx) tuples for batching data
    of the given size and batch_size
    '''
    num_batches = (size + batch_size - 1) // batch_size  # round up
    # num_batches = (size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_tsne(data, y, kind):
    data = np.array(data)
    y = np.array(y)
    X = np.concatenate((data[:]), axis=1)
  
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {} ".format(X.shape[-1], X_tsne.shape[-1]))

    
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # ¹éÒ»»¯
    plt.figure(figsize=(8, 6))
    colmap = plt.cm.get_cmap('jet', kind)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=colmap, s=15)

    plt.xticks([])
    plt.yticks([])
    plt.savefig('./Figure/mnist1_c.pdf', bbox_inches='tight')
    plt.show()