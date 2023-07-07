import pickle
import logging
import datetime
import random
from typing import List

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time as t
import numpy as np
from easydict import EasyDict


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def batch_to_device(tensor_dict: dict, dev):
    for key, obj in tensor_dict.items():
        if torch.is_tensor(obj):
            tensor_dict[key] = obj.to(dev)


def load_pickle(file_path):
    with open(file_path, 'rb') as fr:
        return pickle.load(fr)


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pkl_to_txt(dataset='beauty'):
    data_dir = '../../dataset'
    file_path = os.path.join(data_dir, f'{dataset}/{dataset}_seq.pkl')
    record = load_pickle(file_path)

    # write to xxx.txt
    target_file = os.path.join(data_dir, f'{dataset}/{dataset}_seq.txt')
    with open(target_file, 'w') as fw:
        for seq in record:
            seq = list(map(str, seq))
            seq_str = ' '.join(seq) + '\n'
            fw.write(seq_str)


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def get_activate(act='relu'):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise KeyError(f'Not support current activate function: {act}, please add by yourself.')


class HyperParamDict(EasyDict):
    def __init__(self, description=None):
        super(HyperParamDict, self).__init__({})
        self.description = description
        self.attr_registered = []

    def add_argument(self, param_name, type=object, default=None, action=None, choices=None, help=None):
        param_name = self._parse_param_name(param_name)
        if default and type:
            try:
                default = type(default)
            except Exception:
                assert isinstance(default, type), f'KeyError. Type of param {param_name} should be {type}.'
        if choices:
            assert isinstance(choices, List), f'choices should be a list.'
            assert default in choices, f'KeyError. Please choose {param_name} from {choices}. ' \
                                       f'Now {param_name} = {default}.'
        if action:
            default = self._parse_action(action)
        if help:
            assert isinstance(help, str), f'help should be a str.'
        self.attr_registered.append(param_name)
        self.__setattr__(param_name, default)

    @staticmethod
    def _parse_param_name(param_name: str):
        index = param_name.rfind('-')  # find last pos of -, return -1 on failure
        return param_name[index + 1:]

    @staticmethod
    def _parse_action(action):
        action_infos = action.split('_')
        assert action_infos[0] == 'store' and action_infos[-1] in ['false', 'true'], \
            f"Wrong action format: {action}. Please choose from ['store_false', 'store_true']."
        res = False if action_infos[-1] == 'true' else True
        return res

    def keys(self):
        return self.attr_registered

    def values(self):
        return [self.get(key) for key in self.attr_registered]

    def items(self):
        return [(key, self.get(key)) for key in self.attr_registered]

    def __str__(self):
        info_str = 'HyperParamDict{'
        param_list = []
        for key, value in self.items():
            param_list.append(f'({key}: {value})')
        info_str += ', '.join(param_list) + '}'
        return info_str


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []  # cluster centroids

    def __init_cluster(
            self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        logging.info(f" cluster train iterations: {niter}")
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        """
        Args
            x: batch intent representations of shape [B, D]
        Returns
            seq2cluster: assigned centroid id for each intent of shape [B]
            centroids_assignments: assigned centroid representation for each intent of shape [B, D]
        """
        # self.index.add(x)
        # D : cluster distances of shape [B, 1] I: cluster assignments of shape [B, 1]
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2clusterID = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2clusterID = torch.LongTensor(seq2clusterID).to(self.device)
        centroids_assignments = self.centroids[seq2clusterID]
        return seq2clusterID, centroids_assignments


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


if __name__ == '__main__':
    datasets = ['ml-10m']
    for dataset in datasets:
        pkl_to_txt(dataset)
