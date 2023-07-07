import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            # flatten tensor
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden_view1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


class InfoNCELoss_2(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss, another implementation.
    """

    def __init__(self, temperature, similarity_type, batch_size):
        super(InfoNCELoss_2, self).__init__()
        self.tem = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden1, aug_hidden2):
        """
        Args:
            aug_hidden1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden1.ndim > 2:
            # flatten tensor
            aug_hidden1 = aug_hidden1.view(aug_hidden1.size(0), -1)
            aug_hidden2 = aug_hidden2.view(aug_hidden2.size(0), -1)

        current_batch = aug_hidden1.size(0)
        N = 2 * current_batch
        all_hidden = torch.cat((aug_hidden1, aug_hidden2), dim=0)  # [2*B, D]

        if self.sim_type == 'cos':
            all_hidden = F.normalize(all_hidden)
            sim = torch.mm(all_hidden, all_hidden.T) / self.tem
            # sim = F.cosine_similarity(all_hidden.unsqueeze(1), all_hidden.unsqueeze(0), dim=2) / self.tem
        elif self.sim_type == 'dot':
            sim = torch.mm(all_hidden, all_hidden.T) / self.tem
        else:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        sim_i_j = torch.diag(sim, current_batch)
        sim_j_i = torch.diag(sim, -current_batch)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if self.batch_size != current_batch:
            mask = self.mask_correlated_samples(current_batch)
        else:
            mask = self.mask
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        nce_loss = self.criterion(logits, labels)

        return nce_loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N)).bool()
        mask = mask.fill_diagonal_(0)
        index1 = torch.arange(batch_size) + batch_size
        index2 = torch.arange(batch_size)
        index = torch.cat([index1, index2], 0).unsqueeze(-1)  # [2*B, 1]
        mask = torch.scatter(mask, -1, index, 0)
        return mask


def lalign(x, y, alpha=2):
    return (x - y).norm(dim=-1).pow(alpha).mean()


def lunif(x, t=2):
    sq_dlist = torch.pdist(x, p=2).pow(2)
    return torch.log(sq_dlist.mul(-t).exp().mean() + 1e-6)
