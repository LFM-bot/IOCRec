import numpy as np
import torch


class Metric:
    @staticmethod
    def HIT(prediction, target, k):
        """
        calculate Hit-Ratio (HR) @ k
        :param prediction: [batch, max_k], sorted along dim -1
        :param target: [batch]
        :param k: scalar
        :return: average hit-ratio score among all input data
        """
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()
        hit = hit.sum().item()
        return hit

    @staticmethod
    def NDCG(prediction, target, k):
        """
        calculate Normalized Discounted Cumulative Gain (NDCG) @ k.
        Note that the Ideal Discounted Cumulative Gain (IDCG) is equal to all users, so it can be ignored.
        :param prediction: [batch, max_k], sorted along dim -1
        :param target: [batch]
        :param k: scalar
        :return: average hit-ratio score among all input data
        """
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()  # [batch_size]
        row, col = ((prediction - target) == 0.).nonzero(as_tuple=True)  # [hit_size]
        ndcg = hit.scatter(index=row, src=1. / torch.log2(col + 2).double(), dim=-1)
        ndcg = ndcg.sum().item()
        return ndcg

    @staticmethod
    def MRR(prediction, target, k):
        """
        calculate Mean Reciprocal Rank (MRR) @ k
        :param prediction: [batch, max_k], sorted along dim -1
        :param target: [batch]
        :param k: scalar
        :return: average hit-ratio score among all input data
        """
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()  # [batch_size]
        row, col = ((prediction - target) == 0.).nonzero(as_tuple=True)  # [hit_size]
        mrr = hit.scatter(index=row, src=1. / (col + 1).double(), dim=-1)
        mrr = mrr.sum().item()
        return mrr

    @staticmethod
    def RECALL(prediction, target, k):
        """
        calculate recall @ k, similar to hit-ration under SR (Sequential recommendation) setting
        :param prediction: [batch, max_k], sorted along dim -1
        :param target: [batch]
        :param k: scalar
        :return: average hit-ratio score among all input data
        """
        return Metric.HIT(prediction, target, k)

    @staticmethod
    def process(prediction, target, k):
        if k < prediction.size(-1):
            prediction = prediction[:, :k]  # [batch, k]
        target = target.unsqueeze(-1)  # [batch, 1]
        return prediction, target


if __name__ == '__main__':
    a = torch.arange(12).view(3, -1)
    a[1, -1] = 0
    print(a)
    hit = (a == 0).sum(dim=-1).float()
    hit_index, rank = (a == 0).nonzero(as_tuple=True)
    print(hit_index, rank)
    score = torch.scatter(hit, index=hit_index, src=1. / torch.log2(rank + 2), dim=-1)
    print(score)
    score = score.mean().cpu().numpy()
    print(score)