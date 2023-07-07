import copy
import math
import random
import numpy as np
import torch
import math


class AbstractDataAugmentor:
    def __init__(self, aug_ratio):
        self.aug_ratio = aug_ratio

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        raise NotImplementedError


class CropAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio):
        super(CropAugmentor, self).__init__(aug_ratio)

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        max_len = item_seq.size(-1)
        aug_seq_len = torch.ceil(seq_len * self.aug_ratio).long()
        # get start index
        index = torch.arange(max_len, device=seq_len.device)
        index = index.expand_as(item_seq)
        up_bound = (seq_len - aug_seq_len).unsqueeze(-1)
        prob = torch.zeros_like(item_seq, device=seq_len.device).float()
        prob[index <= up_bound] = 1.
        start_index = torch.multinomial(prob, 1)
        # item indices in subsequence
        gather_index = torch.arange(max_len, device=seq_len.device)
        gather_index = gather_index.expand_as(item_seq)
        gather_index = gather_index + start_index
        max_seq_len = aug_seq_len.unsqueeze(-1)
        gather_index[index >= max_seq_len] = 0
        # augmented subsequence
        aug_seq = torch.gather(item_seq, -1, gather_index).long()
        aug_seq[index >= max_seq_len] = 0

        return aug_seq, aug_seq_len


class MaskDAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio, mask_id=0):
        super(MaskDAugmentor, self).__init__(aug_ratio)
        self.mask_id = mask_id

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        max_len = item_seq.size(-1)
        aug_seq = item_seq.clone()
        aug_seq_len = seq_len.clone()
        # get mask item id
        mask_item_size = math.ceil(max_len * self.aug_ratio)
        prob = torch.ones_like(item_seq, device=seq_len.device).float()
        masked_item_id = torch.multinomial(prob, mask_item_size)
        # mask
        aug_seq = aug_seq.scatter(-1, masked_item_id, self.mask_id)

        return aug_seq, aug_seq_len


class ReorderAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio):
        super(ReorderAugmentor, self).__init__(aug_ratio)

    def transform(self, item_seq, seq_len):
        """
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """
        dev = item_seq.device
        batch_size, max_len = item_seq.shape

        # get start position
        reorder_size = (seq_len * self.aug_ratio).ceil().long().unsqueeze(-1)  # [B, 1]
        position_tensor = torch.arange(max_len).repeat(batch_size, 1).to(dev)  # [B, L]
        sample_prob = (position_tensor <= seq_len.unsqueeze(-1) - reorder_size).bool().float()
        start_index = torch.multinomial(sample_prob, num_samples=1)  # [B, 1]

        # get reorder item mask
        reorder_item_mask = (start_index <= position_tensor) & (position_tensor < start_index + reorder_size)

        # reorder operation
        tmp_reorder_tensor = torch.zeros_like(item_seq).long().to(dev)
        tmp_reorder_tensor[reorder_item_mask] = item_seq[reorder_item_mask]
        rand_index = torch.randperm(max_len)
        tmp_reorder_tensor = tmp_reorder_tensor[:, rand_index]

        # put reordered items back
        aug_item_seq = item_seq.clone()
        aug_item_seq[reorder_item_mask] = tmp_reorder_tensor[tmp_reorder_tensor > 0]

        return aug_item_seq, seq_len


class RepeatAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio):
        super(RepeatAugmentor, self).__init__(aug_ratio)

    def transform_1(self, item_seq, seq_len):
        """
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """
        dev = item_seq.device
        batch_size, max_len = item_seq.shape
        sample_size = math.ceil(max_len * self.aug_ratio)

        # get reordered index
        valid_pos_tensor = torch.arange(max_len).repeat(batch_size, 1).to(dev)
        valid_pos_tensor[item_seq == 0] = -1
        rand_index = torch.randperm(max_len).to(dev)
        rand_pos_tensor = valid_pos_tensor[:, rand_index]
        reordered_index = item_seq.clone()
        reordered_index[reordered_index > 0] = rand_pos_tensor[rand_pos_tensor > 0]

        # break off

        # sample repeat elements
        sample_prob = torch.ones_like(item_seq).float().to(dev)
        repeat_pos = torch.multinomial(sample_prob, num_samples=sample_size)
        sorted_repeat_pos, _ = torch.sort(repeat_pos, dim=-1)
        repeat_element = item_seq.gather(dim=-1, index=sorted_repeat_pos).long()

        # augmented item sequences
        padding_seq = torch.zeros((batch_size, sample_size)).to(dev)
        aug_item_seq = torch.cat([item_seq, padding_seq], dim=-1).long()  # [B, L + L']

        # get insert position mask of sampled item
        valid_pos_tensor = torch.arange(sample_size).unsqueeze(0).to(dev)
        ele_insert_pos = sorted_repeat_pos + valid_pos_tensor
        insert_mask = torch.zeros_like(aug_item_seq).to(item_seq)
        insert_mask = insert_mask.scatter(dim=-1, index=ele_insert_pos, value=1).bool()

        # set elements
        aug_item_seq[insert_mask] = repeat_element.flatten()
        aug_item_seq[~insert_mask] = item_seq.flatten()

        # slice
        full_seq_size = (seq_len == max_len).bool().sum()
        new_seq_len = (aug_item_seq > 0).sum(-1)
        _, sorted_idx = torch.sort(new_seq_len, dim=0, descending=True)
        _, restore_idx = torch.sort(sorted_idx, dim=0)

        sorted_aug_seq = aug_item_seq[sorted_idx]
        full_aug_seq = sorted_aug_seq[:full_seq_size]
        full_aug_seq = full_aug_seq[:, -max_len:]
        non_full_aug_seq = sorted_aug_seq[full_seq_size:]
        non_full_aug_seq = non_full_aug_seq[:, :max_len]
        aug_item_seq = torch.cat([full_aug_seq, non_full_aug_seq], dim=0)

        # restore position
        aug_item_seq = aug_item_seq[restore_idx]
        aug_seq_len = (aug_item_seq > 0).bool().sum(-1)

        return aug_item_seq, aug_seq_len

    def transform(self, item_seq, seq_len):
        """
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """
        dev = item_seq.device
        batch_size, max_len = item_seq.shape
        sample_size = int(item_seq.size(-1) * self.aug_ratio)

        # sample repeat elements
        sample_prob = torch.ones_like(item_seq).float().to(dev)
        repeat_pos = torch.multinomial(sample_prob, num_samples=sample_size)
        sorted_repeat_pos, _ = torch.sort(repeat_pos, dim=-1)
        repeat_element = item_seq.gather(dim=-1, index=sorted_repeat_pos).long()

        # augmented item sequences
        padding_seq = torch.zeros((batch_size, sample_size)).to(dev)
        aug_item_seq = torch.cat([item_seq, padding_seq], dim=-1).long()  # [B, L + L']

        # get insert position mask of sampled item
        position_tensor = torch.arange(sample_size).unsqueeze(0).to(dev)
        ele_insert_pos = sorted_repeat_pos + position_tensor
        insert_mask = torch.zeros_like(aug_item_seq).to(item_seq)
        insert_mask = insert_mask.scatter(dim=-1, index=ele_insert_pos, value=1).bool()

        # set elements
        aug_item_seq[insert_mask] = repeat_element.flatten()
        aug_item_seq[~insert_mask] = item_seq.flatten()

        # slice
        full_seq_size = (seq_len == max_len).bool().sum()
        new_seq_len = (aug_item_seq > 0).sum(-1)
        _, sorted_idx = torch.sort(new_seq_len, dim=0, descending=True)
        _, restore_idx = torch.sort(sorted_idx, dim=0)

        sorted_aug_seq = aug_item_seq[sorted_idx]
        full_aug_seq = sorted_aug_seq[:full_seq_size]
        full_aug_seq = full_aug_seq[:, -max_len:]
        non_full_aug_seq = sorted_aug_seq[full_seq_size:]
        non_full_aug_seq = non_full_aug_seq[:, :max_len]
        aug_item_seq = torch.cat([full_aug_seq, non_full_aug_seq], dim=0)

        # restore position
        aug_item_seq = aug_item_seq[restore_idx]
        aug_seq_len = (aug_item_seq > 0).bool().sum(-1)

        return aug_item_seq, aug_seq_len


class DropAugmentor(AbstractDataAugmentor):
    """
    Torch version of item drop operation.
    """

    def __init__(self, aug_ratio):
        super(DropAugmentor, self).__init__(aug_ratio)

    def transform(self, item_seq, seq_len, drop_prob=None):
        """
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]
        drop_prob: [batch_size, max_len]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """

        dev = item_seq.device
        batch_size, max_len = item_seq.shape
        drop_size = int(item_seq.size(-1) * self.aug_ratio)

        # sample drop item indices
        if drop_prob is None:
            drop_prob = torch.ones_like(item_seq).float().to(dev)
        drop_indices = torch.multinomial(drop_prob, num_samples=drop_size)  # [B, drop_size]

        # fill 0 items
        row_dropped_item_seq = item_seq.scatter(-1, drop_indices, 0).long()
        valid_item_mask = (row_dropped_item_seq > 0).bool()
        dropped_seq_len = valid_item_mask.sum(-1)  # [B]
        position_tensor = torch.arange(max_len).repeat(batch_size, 1).to(dev)  # [B, L]
        valid_pos_mask = (position_tensor < dropped_seq_len.unsqueeze(-1)).bool()

        # post-process
        dropped_item_seq = torch.zeros_like(item_seq).to(dev)
        dropped_item_seq[valid_pos_mask] = row_dropped_item_seq[valid_item_mask]

        # avoid all 0 item
        empty_seq_mask = (dropped_seq_len == 0).bool()
        empty_seq_mask = empty_seq_mask.unsqueeze(-1).repeat(1, max_len)
        empty_seq_mask[:, 1:] = 0
        dropped_item_seq[empty_seq_mask] = item_seq[empty_seq_mask]
        dropped_seq_len = (dropped_item_seq > 0).sum(-1)  # [B]

        return dropped_item_seq, dropped_seq_len


class CauseCropAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio):
        super(CauseCropAugmentor, self).__init__(aug_ratio)

    def transform(self, item_seq, seq_len, critical_mask=None):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        max_len = item_seq.size(-1)
        aug_seq_len = torch.ceil(seq_len * self.aug_ratio).long()
        # get start index
        index = torch.arange(max_len, device=seq_len.device)
        index = index.expand_as(item_seq)
        up_bound = (seq_len - aug_seq_len).unsqueeze(-1)
        prob = torch.zeros_like(item_seq, device=seq_len.device).float()
        prob[index <= up_bound] = 1.
        start_index = torch.multinomial(prob, 1)
        # item indices in subsequence
        gather_index = torch.arange(max_len, device=seq_len.device)
        gather_index = gather_index.expand_as(item_seq)
        gather_index = gather_index + start_index
        max_seq_len = aug_seq_len.unsqueeze(-1)
        gather_index[index >= max_seq_len] = 0
        # augmented subsequence
        aug_seq = torch.gather(item_seq, -1, gather_index).long()
        aug_seq[index >= max_seq_len] = 0

        return aug_seq, aug_seq_len


class CauseReorderAugmentor(AbstractDataAugmentor):
    """
    Torch version.
    """

    def __init__(self, aug_ratio):
        super(CauseReorderAugmentor, self).__init__(aug_ratio)

    def transform(self, item_seq, seq_len):
        """
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """
        dev = item_seq.device
        batch_size, max_len = item_seq.shape

        # get start position
        reorder_size = (seq_len * self.aug_ratio).ceil().long().unsqueeze(-1)  # [B, 1]
        position_tensor = torch.arange(max_len).repeat(batch_size, 1).to(dev)  # [B, L]
        sample_prob = (position_tensor <= seq_len.unsqueeze(-1) - reorder_size).bool().float()
        start_index = torch.multinomial(sample_prob, num_samples=1)  # [B, 1]

        # get reorder item mask
        reorder_item_mask = (start_index <= position_tensor) & (position_tensor < start_index + reorder_size)

        # reorder operation
        tmp_reorder_tensor = torch.zeros_like(item_seq).long().to(dev)
        tmp_reorder_tensor[reorder_item_mask] = item_seq[reorder_item_mask]
        rand_index = torch.randperm(max_len)
        tmp_reorder_tensor = tmp_reorder_tensor[:, rand_index]

        # put reordered items back
        aug_item_seq = item_seq.clone()
        aug_item_seq[reorder_item_mask] = tmp_reorder_tensor[tmp_reorder_tensor > 0]

        return aug_item_seq, seq_len


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        # # add length constraints
        # if len(copied_sequence) < 5:
        #     return copied_sequence

        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length)
        if sub_seq_length < 1:
            return [copied_sequence[min(start_index, len(sequence) - 1)]]
        else:
            cropped_seq = copied_sequence[start_index:start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7, mask_id=0):
        self.gamma = gamma
        self.mask_id = mask_id

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        # # add length constraints
        # if len(copied_sequence) < 5:
        #     return copied_sequence

        mask_nums = int(self.gamma * len(copied_sequence))
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx in mask_idx:
            copied_sequence[idx] = self.mask_id
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        # # add length constraints
        # if len(copied_sequence) < 5:
        #     return copied_sequence

        sub_seq_len = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_len)
        sub_seq = copied_sequence[start_index:start_index + sub_seq_len]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index + sub_seq_len:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class Repeat(object):
    """Randomly repeat p% of items in sequence"""

    def __init__(self, p=0.2, min_rep_size=1):
        self.p = p  # max repeat ratio
        self.min_rep_size = min_rep_size

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        max_repeat_nums = math.ceil(self.p * len(copied_sequence))
        repeat_nums = \
            random.sample([i for i in range(self.min_rep_size, max(self.min_rep_size, max_repeat_nums) + 1)], k=1)[0]
        repeat_idx = random.sample([i for i in range(len(copied_sequence))], k=repeat_nums)
        repeat_idx.sort()
        new_seq = []
        cur_idx = 0
        for i, item in enumerate(copied_sequence):
            new_seq.append(item)
            if cur_idx < len(repeat_idx) and i == repeat_idx[cur_idx]:
                new_seq.append(item)
                cur_idx += 1
        return new_seq


class Drop(object):
    """Randomly repeat p% of items in sequence"""

    def __init__(self, p=0.2):
        self.p = p  # max repeat ratio

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        drop_num = math.floor(self.p * len(copied_sequence))
        drop_idx = random.sample([i for i in range(len(copied_sequence))], k=drop_num)
        drop_idx.sort()
        new_seq = []
        cur_idx = 0
        for i, item in enumerate(copied_sequence):
            if cur_idx < len(drop_idx) and i == drop_idx[cur_idx]:
                cur_idx += 1
                continue
            new_seq.append(item)
        return new_seq


AUGMENTATIONS = {'crop': Crop, 'mask': Mask, 'reorder': Reorder, 'repeat': Repeat, 'drop': Drop}

