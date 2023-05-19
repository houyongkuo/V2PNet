import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import pdist, _hash

def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    if metric == 'cosine':
        return torch.sqrt(2 - 2 * torch.matmul(a, b.T))
    elif metric == 'arccosine':
        return torch.acos(torch.matmul(a, b.T))
    else:
        diffs = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)
        if metric == 'sqeuclidean':
            return torch.sum(diffs ** 2, dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
        elif metric == 'cityblock':
            return torch.sum(torch.abs(diffs), dim=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))

class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.25):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        dist = cdist(anchor, positive, metric=self.metric)
        dist_keypts = np.eye(dist_keypts.shape[0]) * 10 + dist_keypts.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_keypts < self.safe_radius)] += 10
        dist = dist + add_matrix
        return self.calculate_loss(dist, pids)

    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
        # torch.equal is to check whether two tensors have the same size and elements
        # torch.eq is to computes element-wise equality
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
        closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
        # closest_negative_row, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=0)
        # closest_negative = torch.min(closest_negative_col, closest_negative_row)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]
        loss = torch.max(furthest_positive - self.pos_margin, torch.zeros_like(diff)) + torch.max(
            self.neg_margin - closest_negative, torch.zeros_like(diff))

        average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)

        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists


class CircleLoss(nn.Module):
    def __init__(self, dist_type='cosine', log_scale=10, safe_radius=0.10, pos_margin=0.1, neg_margin=1.4):
        super(CircleLoss, self).__init__()
        self.loss = None
        self.log_scale = log_scale
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_margin
        self.neg_optimal = neg_margin
        self.dist_type = dist_type
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        # Calculate distance
        dists = cdist(anchor, positive, metric=self.dist_type)

        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        pos_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        neg_mask = dist_keypts.to(anchor.device) > self.safe_radius

        furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
        closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
        average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

        pos = dists - 1e5 * neg_mask.float()
        pos_weight = (pos - self.pos_optimal).detach()
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)
        lse_positive_row = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-1)
        lse_positive_col = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-2)

        neg = dists + 1e5 * (~neg_mask).float()
        neg_weight = (self.neg_optimal - neg).detach()
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
        lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-1)
        lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-2)

        loss_col = F.softplus(lse_positive_row + lse_negative_row) / self.log_scale
        loss_row = F.softplus(lse_positive_col + lse_negative_col) / self.log_scale
        loss = loss_col + loss_row
        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), dists

class DetLoss(nn.Module):
    def __init__(self, metric='euclidean'):
        super(DetLoss, self).__init__()
        self.metric = metric

    def forward(self, dists, anc_score, pos_score):
        pids = torch.FloatTensor(np.arange(len(anc_score))).to(anc_score.device)
        pos_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        neg_mask = torch.logical_not(pos_mask)

        furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
        closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)

        loss = (furthest_positive - closest_negative) * (anc_score + pos_score).squeeze(-1)
        return torch.mean(loss)

def contrastive_hardest_negative_loss(self,
                                      F0,
                                      F1,
                                      positive_pairs,
                                      num_pos=5192,
                                      num_hn_samples=2048,
                                      thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
    else:
        sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
