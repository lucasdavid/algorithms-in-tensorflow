import tensorflow as tf

from .utils import convert_to_corners


def bbox_iou(b1, b2):
    c1 = convert_to_corners(b1)
    c2 = convert_to_corners(b2)
    lu = tf.maximum(c1[:, None, :2], c2[:, :2])
    rd = tf.minimum(c1[:, None, 2:], c2[:, 2:])
    i = tf.maximum(0.0, rd - lu)
    i_area = i[:, :, 0] * i[:, :, 1]
    b1_area = b1[:, 2] * b1[:, 3]
    b2_area = b2[:, 2] * b2[:, 3]
    union_area = tf.maximum(
        b1_area[:, None] + b2_area - i_area, 1e-8
    )
    return tf.clip_by_value(i_area / union_area, 0.0, 1.0)
