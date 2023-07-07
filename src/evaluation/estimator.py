import re
import torch
from tqdm import tqdm
from src.evaluation.metrics import Metric
from src.utils.utils import batch_to_device


class Estimator:
    def __init__(self, config):
        self.popularity = None
        self.config = config
        self.metrics = config.metric
        self.k_list = config.k
        self.dev = config.device
        self.metric_res_dict = {}
        self.eval_loss = 0.
        self.max_k = max(self.k_list)
        self.split_type = config.split_type
        self.eval_mode = config.eval_mode
        self.neg_size = 0
        if self.eval_mode != 'full':
            self.neg_size = int(re.findall(r'\d+', self.eval_mode)[0])
            self.eval_mode = self.eval_mode[:3]
        self._reset_metrics()
    
    def _reset_metrics(self):
        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] = 0.
        self.eval_loss = 0.
    
    def load_item_popularity(self, pop):
        self.popularity = torch.tensor(pop, dtype=torch.float, device=self.dev)
    
    @torch.no_grad()
    def evaluate(self, eval_loader, model):
        model.eval()
        self._reset_metrics()

        eval_sample_size = len(eval_loader.dataset)
        eval_iter = tqdm(enumerate(eval_loader), total=len(eval_loader))
        eval_iter.set_description(f'do evaluation...')
        for _, batch_dict in eval_iter:
            batch_to_device(batch_dict, self.dev)
            logits = model(batch_dict)
            model_loss = model.get_loss(batch_dict, logits)
            logits = self.neg_sample_select(batch_dict, logits)
            self.calc_metrics(logits, batch_dict['target'])
            self.eval_loss += model_loss.item()

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= float(eval_sample_size)

        eval_loss = self.eval_loss / float(len(eval_loader))

        return self.metric_res_dict, eval_loss

    @torch.no_grad()
    def test(self, test_loader, model):
        model.eval()
        self._reset_metrics()

        test_sample_size = len(test_loader.dataset)
        test_iter = tqdm(enumerate(test_loader), total=len(test_loader))
        test_iter.set_description(f'do test...')
        for _, batch_dict in test_iter:
            batch_to_device(batch_dict, self.dev)
            logits = model(batch_dict)
            logits = self.neg_sample_select(batch_dict, logits)
            self.calc_metrics(logits, batch_dict['target'])

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= float(test_sample_size)
        return self.metric_res_dict

    def calc_metrics(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)  # [batch, max_k]
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        top_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'{metric.upper()}')(top_k_item_sorted, target, k)
                self.metric_res_dict[f'{metric}@{k}'] += score

    def calc_metrics_(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)  # [batch, max_k]
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        max_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        metric_res_dict = {}
        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'{metric.upper()}')(max_k_item_sorted, target, k)
                metric_res_dict[f'{metric}@{k}'] += score

        return metric_res_dict

    def neg_sample_select(self, data_dict, prediction):
        if self.eval_mode == 'full':
            return prediction
        item_seq, target = data_dict['item_seq'], data_dict['target']
        # sample negative items
        target = target.unsqueeze(-1)
        mask_item = torch.cat([item_seq, target], dim=-1)  # [batch, max_len + 1]

        if self.eval_mode == 'uni':
            sample_prob = torch.ones_like(prediction, device=self.dev) / prediction.size(-1)
        elif self.eval_mode == 'pop':
            if self.popularity.size(0) != prediction.size(-1):  # ignore mask item
                self.popularity = torch.cat([self.popularity, torch.zeros((1,)).to(self.dev)], -1)
            sample_prob = self.popularity.unsqueeze(0).repeat(prediction.size(0), 1)
        else:
            raise NotImplementedError('Choose eval_model from [full, popxxx, unixxx]')
        sample_prob = sample_prob.scatter(dim=-1, index=mask_item, value=0.)
        neg_item = torch.multinomial(sample_prob, self.neg_size)  # [batch, neg_size]
        # mask non-rank items
        rank_item = torch.cat([neg_item, target], dim=-1)  # [batch, neg_size + 1]
        mask = torch.ones_like(prediction, device=self.dev).bool()
        mask = mask.scatter(dim=-1, index=rank_item, value=False)
        masked_pred = torch.masked_fill(prediction, mask, 0.)

        return masked_pred

