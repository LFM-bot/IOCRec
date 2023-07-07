import copy
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from src.utils.utils import neg_sample
from src.model.data_augmentation import Crop, Mask, Reorder
from src.model.data_augmentation import AUGMENTATIONS


def load_specified_dataset(model_name, config):
    if model_name in ['CL4SRec', 'ICLRec', 'IOCRec']:
        return CL4SRecDataset
    return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.batch_batch_dict = {}
        self.num_items = config.num_items
        self.config = config
        self.train = train
        self.dataset = config.dataset
        self.max_len = config.max_len
        self.item_seq = data_pair[0]
        self.label = data_pair[1]

    def get_SRtask_input(self, idx):
        item_seq = self.item_seq[idx]
        target = self.label[idx]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    def collate_fn(self, x):
        return self.basic_SR_collate_fn(x)

    def basic_SR_collate_fn(self, x):
        """
        x: [(seq_1, len_1, tar_1), ..., (seq_n, len_n, tar_n)]
        """
        item_seq, seq_len, target = default_collate(x)
        self.batch_batch_dict['item_seq'] = item_seq
        self.batch_batch_dict['seq_len'] = seq_len
        self.batch_batch_dict['target'] = target
        return self.batch_batch_dict


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialDataset, self).__init__(config, data_pair, additional_data_dict, train)


class CL4SRecDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(CL4SRecDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.mask_id = self.num_items
        self.aug_types = config.aug_types
        self.n_views = 2
        self.augmentations = []

        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=True)
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long),
                       torch.tensor(aug_len_1, dtype=torch.long),
                       torch.tensor(aug_len_2, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)

        item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = default_collate(x)

        self.batch_batch_dict['item_seq'] = item_seq
        self.batch_batch_dict['seq_len'] = seq_len
        self.batch_batch_dict['target'] = target
        self.batch_batch_dict['aug_seq_1'] = aug_seq_1
        self.batch_batch_dict['aug_seq_2'] = aug_seq_2
        self.batch_batch_dict['aug_len_1'] = aug_len_1
        self.batch_batch_dict['aug_len_2'] = aug_len_2

        return self.batch_batch_dict


class MISPPretrainDataset(Dataset):
    """
    Masked Item & Segment Prediction (MISP)
    """

    def __init__(self, config, data_pair, additional_data_dict=None):
        self.mask_id = config.num_items
        self.mask_ratio = config.mask_ratio
        self.num_items = config.num_items + 1
        self.config = config
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.long_sequence = []

        for seq in self.item_seq:
            self.long_sequence.extend(seq)

    def __len__(self):
        return len(self.item_seq)

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # Masked Item Prediction
        masked_item_sequence = []
        neg_items = []
        pos_items = sequence

        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))

        assert len(masked_item_sequence) == len(sequence)
        assert len(pos_items) == len(sequence)
        assert len(neg_items) == len(sequence)

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                             start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len
        neg_items = neg_items + [0] * pad_len
        masked_segment_sequence = masked_segment_sequence + [0] * pad_len
        pos_segment = pos_segment + [0] * pad_len
        neg_segment = neg_segment + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long))
        return cur_tensors

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        masked_item_sequence, pos_items, neg_items, \
        masked_segment_sequence, pos_segment, neg_segment = tensor_list

        tensor_dict['masked_item_sequence'] = masked_item_sequence
        tensor_dict['pos_items'] = pos_items
        tensor_dict['neg_items'] = neg_items
        tensor_dict['masked_segment_sequence'] = masked_segment_sequence
        tensor_dict['pos_segment'] = pos_segment
        tensor_dict['neg_segment'] = neg_segment

        return tensor_dict


class MIMPretrainDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None):
        self.config = config
        self.aug_types = config.aug_types
        self.mask_id = config.num_items

        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.n_views = 2
        self.augmentations = []
        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

    def __getitem__(self, index):
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=False)
        item_seq = self.item_seq[index]
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        aug_seq_tensors = (torch.tensor(aug_seq_1, dtype=torch.long),
                           torch.tensor(aug_seq_2, dtype=torch.long),
                           torch.tensor(aug_len_1, dtype=torch.long),
                           torch.tensor(aug_len_2, dtype=torch.long))

        return aug_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = tensor_list

        tensor_dict['aug_seq_1'] = aug_seq_1
        tensor_dict['aug_seq_2'] = aug_seq_2
        tensor_dict['aug_len_1'] = aug_len_1
        tensor_dict['aug_len_2'] = aug_len_2

        return tensor_dict


class PIDPretrainDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None):
        self.num_items = config.num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.pseudo_ratio = config.pseudo_ratio

    def __getitem__(self, index):
        item_seq = self.item_seq[index]
        pseudo_seq = []
        target = []

        for item in item_seq:
            if random.random() < self.pseudo_ratio:
                pseudo_item = neg_sample(item_seq, self.num_items)
                pseudo_seq.append(pseudo_item)
                target.append(0)
            else:
                pseudo_seq.append(item)
                target.append(1)

        pseudo_seq = pseudo_seq[-self.max_len:]
        target = target[-self.max_len:]

        pseudo_seq = pseudo_seq + [0] * (self.max_len - len(pseudo_seq))
        target = target + [0] * (self.max_len - len(target))
        assert len(pseudo_seq) == self.max_len
        assert len(target) == self.max_len
        pseudo_seq_tensors = (torch.tensor(pseudo_seq, dtype=torch.long),
                              torch.tensor(target, dtype=torch.float))

        return pseudo_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        pseudo_seq, target = tensor_list

        tensor_dict['pseudo_seq'] = pseudo_seq
        tensor_dict['target'] = target

        return tensor_dict


