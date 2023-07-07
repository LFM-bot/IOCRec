import codecs
import copy
import logging
import dgl
import numpy as np
import torch
import os
from scipy import sparse
from src.utils.utils import save_pickle, load_pickle


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model
        self.dataset = config.dataset
        self.data_aug = config.data_aug

        self.use_tar_seq = False
        self.tar_seq_len = 1
        if self.model_name in ['KERL', 'MELOD']:
            self.use_tar_seq = True
            self.tar_seq_len = config.episode_len
        self.filter_len = config.seq_filter_len
        self.filter_target = config.if_filter_target
        self.item_id_pad = 1  # increase item id: x -> x + 1
        self.sep = config.separator
        self.graph_type_list = [g_type.upper() for g_type in config.graph_type]
        self.valid_graphs = ['GGNN', 'BIPARTITE', 'TRANSITION', 'HYPER']

        self.data_path = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.kg_map = None
        self.do_data_split = True
        self.popularity = None

        self.split_type = config.split_type
        self.split_mode = config.split_mode
        self.eval_ratio = 0.2  # default

        # data statistic
        self.max_item = 0
        self.num_items = 0  # with padding item 0
        self.num_users = 0
        self.seq_avg_len = 0.
        self.total_seq_len = 0.
        self.density = 0.
        self.sparsity = 0.

        self._init_data_processer()

    def _init_data_processer(self):
        if self.split_mode == 'PS':
            self.do_data_split = False
        elif 'LS_R' == self.split_mode.split('@')[0]:
            self.eval_ratio = float(self.split_mode.split('@')[-1])
            self.split_mode = 'LS_R'
        self._set_data_path()

    def prepare_data(self):
        if self.do_data_split:
            seq_data_list = self._load_row_data()
            self._train_test_split(seq_data_list)
        else:  # load pre-split data
            self._load_pre_split_data()

        data_dict = {'train': self.train_data,
                     'eval': self.eval_data,
                     'test': self.test_data,
                     'raw_train': self.row_train_data}

        extra_data_dict = self._prepare_additional_data()

        return data_dict, extra_data_dict

    def _prepare_additional_data(self):
        additional_data_dict = {}
        if self.config.model_type.lower() == 'graph':
            specified_graph_dict = self.prepare_specified_graph()
            if specified_graph_dict is not None:
                additional_data_dict['graph_dict'] = specified_graph_dict
        if self.config.model_type.lower() == 'knowledge':
            if self.config.kg_data_type == 'pretrain':
                kg_map = self.prepare_kg_map().float().to(self.config.device)
                additional_data_dict['kg_map'] = kg_map
            elif self.config.kg_data_type == 'jointly_train':
                triple_list, relation_tph, relation_hpt = self.load_kg_triplet(entity_id_shift=1)
                additional_data_dict['triple_list'] = triple_list
                additional_data_dict['relation_tph'] = relation_tph
                additional_data_dict['relation_hpt'] = relation_hpt
            else:
                pass
        if self.model_name in ['DuoRec']:
            train_target_item = self.train_data[1]
            same_target_index = self.semantic_augmentation(train_target_item)
            additional_data_dict['same_target_index'] = same_target_index
        elif self.model_name in ['KGSCL']:
            kg_relation_dict = load_pickle(os.path.join(self.data_path, 'kg_relation_set.pkl'))
            co_occurrence_dict = self.calc_co_occurrence(kg_relation_dict)
            additional_data_dict['kg_relation_dict'] = kg_relation_dict
            additional_data_dict['co_occurrence_dict'] = co_occurrence_dict

        return additional_data_dict

    def semantic_augmentation(self, target_item):
        target_item = np.array(target_item, dtype=np.long)
        aug_path = self.data_path + '/semantic_augmentation.pkl'
        target_deduplicated = set(target_item)
        if os.path.exists(aug_path):
            same_target_index = load_pickle(aug_path)
        else:
            same_target_index = {}
            for _, item_id in enumerate(target_deduplicated):
                all_index_same_id_w_self = np.where(target_item == item_id)[
                    0].tolist()  # all index of a specific item id with self item, including itself
                if item_id not in same_target_index.keys():
                    same_target_index[item_id] = []
                same_target_index[item_id].extend(all_index_same_id_w_self)
            save_pickle(same_target_index, aug_path)
        return same_target_index

    def calc_co_occurrence(self, kg_relation_dict):
        dict_path = self.data_path + '/co_occurrence.pkl'
        if os.path.exists(dict_path):
            co_occurrence_dict = load_pickle(dict_path)
            return co_occurrence_dict
        else:
            co_occurrence_all_data = {}
            for item_seq, target in zip(*self.row_train_data):
                for i in range(len(item_seq)):
                    for j in range(i + 1, len(item_seq)):
                        item_1, item_2 = item_seq[i], item_seq[j]
                        co_item_pair = (item_1, item_2) if item_1 < item_2 else (item_2, item_1)
                        co_item_pair = f'{co_item_pair[0]}-{co_item_pair[1]}'
                        if co_item_pair not in co_occurrence_all_data.keys():
                            co_occurrence_all_data[co_item_pair] = 0.
                        co_occurrence_all_data[co_item_pair] += 1.
                # train target item
                for item in item_seq:
                    co_item_pair = (item, target) if item < target else (target, item_seq)
                    co_item_pair = f'{co_item_pair[0]}-{co_item_pair[1]}'
                    if co_item_pair not in co_occurrence_all_data.keys():
                        co_occurrence_all_data[co_item_pair] = 0.
                    co_occurrence_all_data[co_item_pair] += 1.

            # calc co-occurrence for items with kg relations
            co_occurrence_dict = {}
            for item in kg_relation_dict.keys():
                shifted_item = item + 1
                co_occurrence_dict[item] = {'s': [], 'c': []}

                # frequency of substitute items
                substitute_freq_list = []
                substitute_item_list = kg_relation_dict[item]['s']
                if len(substitute_item_list) > 0:
                    for sub_item in substitute_item_list:
                        shifted_sub_item = sub_item + 1
                        co_item_pair = (shifted_item, shifted_sub_item) if item < sub_item else (
                            shifted_sub_item, shifted_item)
                        co_item_pair = f'{co_item_pair[0]}-{co_item_pair[1]}'
                        if co_item_pair not in co_occurrence_all_data.keys():
                            substitute_freq_list.append(1.)
                        else:
                            substitute_freq_list.append(1. + float(co_occurrence_all_data[co_item_pair]))
                    sum_freq = sum(substitute_freq_list)
                    substitute_freq_list = [freq / sum_freq for freq in substitute_freq_list]
                assert len(substitute_freq_list) == len(substitute_item_list)
                co_occurrence_dict[item]['s'] = substitute_freq_list

                # frequency of complement items
                complement_freq_list = []
                complement_item_list = kg_relation_dict[item]['c']
                if len(complement_item_list) > 0:
                    for sub_item in complement_item_list:
                        shifted_sub_item = sub_item + 1
                        co_item_pair = (shifted_item, shifted_sub_item) if item < sub_item else (
                            shifted_sub_item, shifted_item)
                        co_item_pair = f'{co_item_pair[0]}-{co_item_pair[1]}'
                        if co_item_pair not in co_occurrence_all_data.keys():
                            complement_freq_list.append(1.)
                        else:
                            complement_freq_list.append(1. + co_occurrence_all_data[co_item_pair])
                    sum_freq = sum(complement_freq_list)
                    complement_freq_list = [freq / sum_freq for freq in complement_freq_list]
                assert len(complement_freq_list) == len(complement_item_list)
                co_occurrence_dict[item]['c'] = complement_freq_list

            save_pickle(co_occurrence_dict, dict_path)
            return co_occurrence_dict

    def load_kg_triplet(self, entity_id_shift=1):
        """
        entity_id_shift: entity_id += 1

        return:
        triple_list: list of kg triplets
        relation_tph: dict, tail number for each head in relation r
        relation_hpt: dict, head number of each tail in relation r
        """
        # required kg files
        train_triplet = f'{self.data_path}/kg_data/train.txt'
        entity2id = f'{self.data_path}/kg_data/entities.dict'
        relation2id = f'{self.data_path}/kg_data/relations.dict'

        entities2id = {}
        relations2id = {}
        relation_tph = {}
        relation_hpt = {}

        entity = []
        relation = []
        # read entity and relation sets
        with open(entity2id, 'r') as f1, open(relation2id, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                entities2id[line[0]] = int(line[1]) + entity_id_shift
                entity.append(int(line[1]))

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                relations2id[line[0]] = int(line[1])
                relation.append(int(line[1]))

        triple_list = []
        relation_head = {}
        relation_tail = {}

        # read kg triplets, convert entity name to entity id
        with codecs.open(train_triplet, 'r') as f:
            content = f.readlines()
            for line in content:
                triple = line.strip().split("\t")
                if len(triple) != 3:
                    continue

                h_ = int(entities2id[triple[0]])
                r_ = int(relations2id[triple[1]])
                t_ = int(entities2id[triple[2]])

                triple_list.append([h_, r_, t_])
                if r_ in relation_head:
                    if h_ in relation_head[r_]:
                        relation_head[r_][h_] += 1
                    else:
                        relation_head[r_][h_] = 1
                else:
                    relation_head[r_] = {}
                    relation_head[r_][h_] = 1

                if r_ in relation_tail:
                    if t_ in relation_tail[r_]:
                        relation_tail[r_][t_] += 1
                    else:
                        relation_tail[r_][t_] = 1
                else:
                    relation_tail[r_] = {}
                    relation_tail[r_][t_] = 1

        for r_ in relation_head:
            sum1, sum2 = 0, 0
            for head in relation_head[r_]:
                sum1 += 1
                sum2 += relation_head[r_][head]
            tph = sum2 / sum1
            relation_tph[r_] = tph

        for r_ in relation_tail:
            sum1, sum2 = 0, 0
            for tail in relation_tail[r_]:
                sum1 += 1
                sum2 += relation_tail[r_][tail]
            hpt = sum2 / sum1
            relation_hpt[r_] = hpt

        print("Complete load. entity : %d , relation : %d , train triple : %d" % (
            len(entity), len(relation), len(triple_list)))

        # update config
        self.config.entity_num = len(entity) + 1  # add padding item 0
        self.config.relation_num = len(relation)

        # tph: tails per head, hpt: heads per tail
        return triple_list, relation_tph, relation_hpt

    def _set_data_path(self):
        # find file path
        cur_path = os.path.abspath(__file__)
        root = '\\'.join(cur_path.split('\\')[:-3])
        self.data_path = os.path.join(root, f'dataset/{self.dataset}')

    def _read_seq_data(self, file_path):
        # read data file
        data_list = []
        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                item_seq = list(map(int, line.strip().split(self.sep)))
                # remove target items
                if self.filter_target:
                    item_seq = self._filter_target(item_seq)
                # drop too short sequence
                if len(item_seq) < self.filter_len:
                    continue
                item_seq = [item + self.item_id_pad for item in item_seq]  # shift item id x to x + 1
                # statistic
                self.max_item = max(self.max_item, max(item_seq))
                self.total_seq_len += float(len(item_seq))
                self.num_users += 1
                data_list.append(item_seq)
        return data_list

    def _load_row_data(self):
        """
        load total data sequences
        """
        file_path = os.path.join(self.data_path, f'{self.dataset}.seq')
        seq_data_list = self._read_seq_data(file_path)
        self._set_statistic(seq_data_list)

        return seq_data_list

    def _set_statistic(self, seq_data_list=None):
        self.seq_avg_len = round(float(self.total_seq_len) / self.num_users, 4)
        self.density = round(float(self.total_seq_len) / self.num_users / self.max_item, 4)
        self.sparsity = 1 - self.density
        self.num_users = int(self.num_users)
        self.num_items = int(self.max_item + 1)  # with padding item 0

        # calculate popularity
        self.popularity = [0. for _ in range(self.num_items)]
        for item_seq in seq_data_list:
            for item in item_seq:
                self.popularity[item] += 1.
        self.popularity = [p / self.total_seq_len for p in self.popularity]

    def _load_pre_split_data(self):
        """
        load data after split, xx.train, xx.eval, xx.test
        """
        # load xx.train, xx.eval
        train_file = os.path.join(self.data_path, f'{self.dataset}.train')
        eval_file = os.path.join(self.data_path, f'{self.dataset}.eval')

        train_data_list = self._read_seq_data(train_file)
        eval_data_list = self._read_seq_data(eval_file)

        train_x = [seq[:-1] for seq in train_data_list if len(seq) > 1]
        train_y = [seq[-1] for seq in train_data_list if len(seq) > 1]
        eval_x = [seq[:-1] for seq in eval_data_list if len(seq) > 1]
        eval_y = [seq[-1] for seq in eval_data_list if len(seq) > 1]

        self.row_train_data = copy.deepcopy(train_x), copy.deepcopy(train_y)
        # training data augmentation
        self._data_augmentation(train_x, train_y)

        # load xx.test
        if self.split_type == 'valid_and_test':
            test_file = os.path.join(self.data_path, f'{self.dataset}.test')
            test_data_list = self._read_seq_data(test_file)
            test_x = [seq[:-1] for seq in test_data_list if len(seq) > 1]
            test_y = [seq[-1] for seq in test_data_list if len(seq) > 1]
            self.test_data = (test_x, test_y)
            test_seq = [seq + [target] for seq, target in zip(test_x, test_y)]

        self.train_data = (train_x, train_y)
        self.eval_data = (eval_x, eval_y)

        # gather all sequences
        all_data_list = [seq + [target] for seq, target in zip(train_x, train_y)]
        eval_seq = [seq + [target] for seq, target in zip(eval_x, eval_y)]
        all_data_list.extend(eval_seq)
        if self.split_type == 'valid_and_test':
            all_data_list.extend(test_seq)

        self._set_statistic(all_data_list)

    def _train_test_split(self, seq_data_list):
        if self.split_type == 'valid_only':
            train_x, train_y, eval_x, eval_y = self._leave_one_out_split(seq_data_list)
        else:  # valid and test
            if self.split_mode == 'LS':
                train_x = [item_seq[:-3] for item_seq in seq_data_list if len(item_seq) > 3]
                train_y = [item_seq[-3] for item_seq in seq_data_list if len(item_seq) > 3]
                eval_x = [item_seq[:-2] for item_seq in seq_data_list if len(item_seq) > 2]
                eval_y = [item_seq[-2] for item_seq in seq_data_list if len(item_seq) > 2]
                test_x = [item_seq[:-1] for item_seq in seq_data_list if len(item_seq) > 1]
                test_y = [item_seq[-1] for item_seq in seq_data_list if len(item_seq) > 1]
            else:  # LS_R
                train_x, train_y, test_x, test_y = self._leave_one_out_split(seq_data_list)
                # split eval and test data by ratio
                eval_x, eval_y, test_x, test_y = self._split_by_ratio(test_x, test_y)
            self.test_data = (test_x, test_y)

        self.row_train_data = (copy.deepcopy(train_x), copy.deepcopy(train_y))
        # training data augmentation
        self._data_augmentation(train_x, train_y)

        self.train_data = (train_x, train_y)
        self.eval_data = (eval_x, eval_y)

    def _leave_one_out_split(self, seq_data):
        train_x = [item_seq[:-2] for item_seq in seq_data if len(item_seq) > 2]
        train_y = [item_seq[-2] for item_seq in seq_data if len(item_seq) > 2]
        eval_x = [item_seq[:-1] for item_seq in seq_data if len(item_seq) > 1]
        eval_y = [item_seq[-1] for item_seq in seq_data if len(item_seq) > 1]
        return train_x, train_y, eval_x, eval_y

    def prepare_kg_map(self):
        kg_path = os.path.join(self.data_path, f'{self.dataset}_kg.npy')
        kg_map_np = np.load(kg_path)
        zero_kg_emb = np.zeros((1, kg_map_np.shape[-1]))
        kg_map_np_pad = np.concatenate([zero_kg_emb, kg_map_np], axis=0)
        return torch.from_numpy(kg_map_np_pad)

    def prepare_specified_graph(self):
        assert isinstance(self.graph_type_list, list), f'graph_type should be a list.'
        graph_dict = {}
        for g_type in self.graph_type_list:
            if g_type not in self.valid_graphs:
                raise KeyError(f'Invalid graph type:{self.graph_type_list}. Choose from {self.valid_graphs}')
            if g_type == 'GGNN':
                continue  # session graph will be constructed in dataset
            graph_dict[g_type] = getattr(self, f'prepare_{g_type.lower()}_graph')()
        if len(graph_dict) == 0:
            return None
        return graph_dict

    def prepare_bipartite_graph(self, bidirectional=True):
        u_i_edges = []
        i_u_edges = []

        for user, item_list in enumerate(self.row_train_data[0]):
            for item in item_list:
                u_i_edges.append((user, item))
                if bidirectional:  # bidirectional graph
                    i_u_edges.append((item, user))
        num_nodes_dict = {'user': len(self.row_train_data[0]), 'item': self.num_items}

        logging.info('loading graph...')
        if bidirectional:
            bipartite = dgl.heterograph({
                ('user', 'contact', 'item'): u_i_edges,
                ('item', 'contact_by', 'user'): i_u_edges
            }, num_nodes_dict=num_nodes_dict)
        else:
            bipartite = dgl.heterograph({
                ('user', 'contact', 'item'): u_i_edges
            }, num_nodes_dict=num_nodes_dict)

        logging.info(bipartite)
        return bipartite

    def prepare_transition_graph(self):
        """
        Returns:
            adj (sparse matrix): adjacency matrix of item transition graph
        """
        src, des = [], []
        for item_seq in self.row_train_data[0]:
            for i in range(len(item_seq) - 1):
                src.append(item_seq[i])
                des.append(item_seq[i + 1])
        adj = sparse.coo_matrix((np.ones(len(src), dtype=float), (src, des)), shape=(self.num_items, self.num_items))

        # adj = sparse.load_npz(f'{self.data_path}/global_adj.npz')

        # frequency divide in degree of destination node as edge weight
        # in_degree = adj.sum(0).reshape(1, -1)
        # in_degree[in_degree == 0] = 1
        # adj = adj.multiply(1 / in_degree)

        return adj

    def prepare_hyper_graph(self):
        """
        Returns:
            incidence_matrix (sparse matrix): incidence matrix for session hyper-graph
        """
        # incidence_matrix = sparse.load_npz(f'{self.data_path}/global_adj.npz')
        # return incidence_matrix

        nodes, edges = [], []
        for edge, item_seq in enumerate(self.row_train_data[0]):
            for item in item_seq:
                nodes.append(item)
                edges.append(edge)
        incidence_matrix = sparse.coo_matrix((np.ones(len(nodes), dtype=float), (nodes, edges)))

        return incidence_matrix

    def data_log_verbose(self, order):

        logging.info(f'[{order}] Data Statistic '.ljust(47, '-'))
        logging.info(f'dataset: {self.dataset}')
        logging.info(f'user number: {self.num_users}')
        logging.info(f'item number: {self.max_item}')
        logging.info(f'average seq length: {self.seq_avg_len}')
        logging.info(f'density: {self.density} sparsity: {self.sparsity}')
        if self.data_aug:
            logging.info(f'data after augmentation:')
            if self.split_type == 'valid_only':
                logging.info(f'train samples: {len(self.train_data[0])}\teval samples: {len(self.eval_data[0])}')
            else:
                logging.info(f'train samples: {len(self.train_data[0])}\teval samples: {len(self.eval_data[0])}\ttest '
                             f'samples: {len(self.test_data[0])}')
        else:
            logging.info(f'data without augmentation:')
            if self.split_type == 'valid_only':
                logging.info(f'train samples: {len(self.train_data[0])}\teval samples: {len(self.eval_data[0])}')
            else:
                logging.info(f'train samples: {len(self.train_data[0])}\teval samples: {len(self.eval_data[0])}\ttest '
                             f'samples: {len(self.test_data[0])}')

    def _filter_target(self, item_seq):
        target = item_seq[-1]
        item_seq = list(filter(lambda x: x != target, item_seq[:-1]))
        item_seq.append(target)
        return item_seq

    def _split_by_ratio(self, test_x, test_y):
        """
        random split by specified ratio
        """
        eval_size = int(len(test_y) * self.eval_ratio)
        index = np.arange(len(test_y))
        np.random.shuffle(index)

        eval_index = index[:eval_size]
        test_index = index[eval_size:]

        eval_x = [test_x[i] for i in eval_index]
        eval_y = [test_y[i] for i in eval_index]

        test_x = [test_x[i] for i in test_index]
        test_y = [test_y[i] for i in test_index]

        return eval_x, eval_y, test_x, test_y

    def _data_augmentation(self, train_x, train_y):
        if not self.data_aug:
            return
        if not self.use_tar_seq:
            aug_train_x = [item_seq[:last] for item_seq in train_x for last in range(1, len(item_seq))]
            aug_train_y = [item_seq[nextI] for item_seq in train_x for nextI in range(1, len(item_seq))]
        else:
            aug_train_x = [item_seq[:last] for item_seq in train_x for last in range(1, len(item_seq))]
            aug_train_y = [item_seq[nextI: nextI + self.tar_seq_len] for item_seq in train_x for nextI in
                           range(1, len(item_seq))]
        train_x.extend(aug_train_x)
        train_y.extend(aug_train_y)
