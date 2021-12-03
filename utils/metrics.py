import torch
import numpy as np
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


class MultiThresholdMetric(object):
    def __init__(self, threshold):

        self._thresholds = threshold[ :, None, None, None, None] # [Tresh, B, C, H, W]
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def _normalize_dimensions(self):
        ''' Converts y_truth, y_label and threshold to [B, Thres, C, H, W]'''
        # Naively assume that all of existing shapes of tensors, we transform [B, H, W] -> [B, Thresh, C, H, W]
        self._thresholds = self._thresholds[ :, None, None, None, None] # [Tresh, B, C, H, W]
        # self._y_pred = self._y_pred[None, ...]  # [B, Thresh, C, ...]
        # self._y_true = self._y_true[None,:, None, ...] # [Thresh, B,  C, ...]

    def add_sample(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.bool()[None,...] # [Thresh, B,  C, ...]
        y_pred = y_pred[None, ...]  # [Thresh, B, C, ...]
        y_pred_offset = (y_pred - self._thresholds + 0.5).round().bool()

        self.TP += (y_true & y_pred_offset).sum(dim=self._data_dims).float()
        self.TN += (~y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FP += (y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FN += (~y_true & y_pred_offset).sum(dim=self._data_dims).float()

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall

    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom


def true_pos(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim)  # Only sum along H, W axis, assuming no C


def false_pos(y_true, y_pred, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def false_neg(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def precision(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FP = false_pos(y_true, y_pred, dim)
    denom = TP + FP
    denom = torch.clamp(denom, 10e-05)
    return TP / denom


def recall(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FN = false_neg(y_true, y_pred, dim)
    denom = TP + FN
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom


def f1_score(gts:torch.Tensor, preds:torch.Tensor, multi_threashold_mode=False, dim=(-1, -2)):
    # FIXME Does not operate proper
    gts = gts.float()
    preds = preds.float()

    if multi_threashold_mode:
        gts = gts[:, None, ...] # [B, Thresh, ...]
        gts = gts.expand_as(preds)

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


def f1_score_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    p = precsision_from_prob(y_prob, y_true, threshold=threshold)
    r = recall_from_prob(y_prob, y_true, threshold=threshold)
    return 2 * (p * r) / (p + r + sys.float_info.epsilon)


def true_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    return np.sum(np.logical_and(y_pred, y_true))


def false_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    return np.sum(np.logical_and(y_pred, np.logical_not(y_true)))


def false_negatives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    return np.sum(np.logical_and(np.logical_not(y_pred), y_true))


def precsision_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fp)


def recall_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fn + sys.float_info.epsilon)


def root_mean_square_error(y_pred: np.ndarray, y_true: np.ndarray):
    return np.sqrt(np.sum(np.square(y_pred - y_true)) / np.size(y_true))
