import tensorflow as tf


def dice_coefficient(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def true_positives(y_true, y_pred):
    return tf.math.count_nonzero(tf.math.logical_and(y_pred, y_true))


def false_positives(y_true, y_pred):
    return tf.math.count_nonzero(tf.math.logical_and(y_pred, tf.math.logical_not(y_true)))


def false_negatives(y_true, y_pred):
    return tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(y_pred), y_true))


def precision(y_true, y_pred):
    TP = true_positives(y_true, y_pred)
    FP = false_positives(y_true, y_pred)
    return 0 if (TP + FP) == 0 else TP / (TP + FP)


def recall(y_true, y_pred):
    TP = true_positives(y_true, y_pred)
    FN = false_negatives(y_true, y_pred)
    return 0 if (TP + FN) == 0 else TP / (TP + FN)


def f1_score(y_true, y_pred):
    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 0 if (pr + re) == 0 else (2 * pr * re) / (pr + re)


