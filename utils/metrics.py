import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.functional as F


def calculate_acc(predict, labels):
    pred_labels = torch.max(predict, dim=1)[1].int()
    return torch.sum(pred_labels == labels.int()) * 100 / predict.shape[0]


def calculate_iou_single_shape(predict, labels, n_parts):
    pred_labels = torch.max(predict, dim=1)[1]
    Confs = confusion_matrix(labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), np.arange(n_parts))

    # Objects IoU
    IoUs = IoU_from_confusions(Confs)
    return IoUs


def calculate_iou(predict, labels, stack_lengths, n_parts):
    start_ind = 0
    iou_list = []
    for length in stack_lengths:
        iou = calculate_iou_single_shape(predict[start_ind:start_ind + length], labels[start_ind:start_ind + length],
                                         n_parts)
        iou_list.append(iou)
        start_ind += length
    iou_list = np.array(iou_list)
    return np.array(iou_list).mean(axis=0) * 100


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def batch_rotation_error(rots1, rots2):
    r"""
  arccos( (tr(R_1^T R_2) - 1) / 2 )
  rots1: B x 3 x 3 or B x 9
  rots1: B x 3 x 3 or B x 9
  """
    assert len(rots1) == len(rots2)
    trace_r1Tr2 = (rots1.reshape(-1, 9) * rots2.reshape(-1, 9)).sum(1)
    side = (trace_r1Tr2 - 1) / 2
    return torch.acos(torch.clamp(side, min=-0.999, max=0.999))


def batch_translation_error(trans1, trans2):
    r"""
  trans1: B x 3
  trans2: B x 3
  """
    assert len(trans1) == len(trans2)
    return torch.norm(trans1 - trans2, p=2, dim=1, keepdim=False)


# FCGF
def eval_metrics(output, target):
    output = (F.sigmoid(output) > 0.5).cpu().data.numpy()
    target = target.cpu().data.numpy()
    return np.linalg.norm(output - target)


def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
    xyz0_est = xyz0 @ est[:3, :3].t() + est[:3, 3]
    xyz0_gth = xyz0 @ gth[:3, :3].t() + gth[:3, 3]
    dists = torch.clamp(torch.sqrt(((xyz0_est - xyz0_gth).pow(2)).sum(1)), max=max_dist)
    if weight is not None:
        dists = weight * dists
    return dists.mean()


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


def get_loss_fn(loss):
    if loss == 'corr_dist':
        return corr_dist
    else:
        raise ValueError(f'Loss {loss}, not defined')


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec
