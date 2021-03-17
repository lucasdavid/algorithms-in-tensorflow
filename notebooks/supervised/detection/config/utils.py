import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            initial_learning_rate,
            first_decay_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=1e-6,
            name=None):
        super(CosineDecayRestarts, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.math.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                        tf.math.log(t_mul))

                    sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

                else:
                    i_restart = tf.math.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.math.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul ** i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + tf.math.cos(
                tf.constant(math.pi) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha

            return tf.math.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name
        }


# region Data Utils

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    input box : ymin / h, xmin / w, ymax / h, xmax / w
    output    : (ymin+ymax / 2 h, xmin+xmax / 2 w,
                 ymax-ymin / h, xmax-xmin / w)
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1)


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1)


# endregion

def visualize_detections(
        image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=(0, 0, 1)
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax
