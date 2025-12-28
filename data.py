import numpy as np
import scipy.io as sio
import utils
import torch
from scipy.sparse import issparse
import gzip
from sklearn import preprocessing
def get_data(path):
    data = sio.loadmat(path)
    view_size = len(data['X'][0])
    print(view_size)
    data_list = []
    for i in range(view_size):
        x_train, y_train, x_test, y_test = load_data(data, i)
        ret = (x_train, y_train, x_test, y_test)
        data_list.append(ret)

    return data_list


def load_data(data, view):
    X = data['X'][0]
    x = X[view]
    # print(type(x))
    if issparse(x):
        x = np.asarray(x.todense())
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    # x = preprocessing.scale(x, axis=0, with_std=True)   # 规范化

    # x = preprocessing.normalize(x.T, norm="max")
    # x = x.T

    # x = x.T
    # sc = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # sc.fit(x)
    # x = sc.transform(x)
    # x = x.T

    y = np.squeeze(data['Y'])
    print(np.min(y))
    if np.min(y) == 1:
        y = y-1

    data_size = x.shape[0]
    train_index, test_index = utils.random_index(data_size, int(data_size * 0.8), 1)
    test_set_x = torch.tensor(x[test_index].astype(np.float32))
    test_set_y = torch.tensor(y[test_index])
    train_set_x = torch.tensor(x[train_index].astype(np.float32))
    train_set_y = torch.tensor(y[train_index])

    
    return train_set_x, train_set_y, test_set_x, test_set_y

def load_data_t(X, y):

    data_size = X.shape[0]
    train_index, test_index = utils.random_index(data_size, int(data_size * 0.8), 1)
    test_set_x = X[test_index]
    test_set_y = y[test_index]
    train_set_x = X[train_index]
    train_set_y = y[train_index]

    
    return train_set_x, train_set_y, test_set_x, test_set_y


