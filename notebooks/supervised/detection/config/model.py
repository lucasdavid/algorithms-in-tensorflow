import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D
from tensorflow.python.keras.layers import Activation


def get_backbone(layers=('conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out')):
    resnet50 = tf.keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])

    return Model(inputs=[resnet50.inputs],
                 outputs=[resnet50.get_layer(l).output for l in layers],
                 name='resnet50_backbone')


class FeaturePyramid(Layer):
    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name='FeaturePyramid', **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = Conv2D(256, 1, 1, 'same')
        self.conv_c4_1x1 = Conv2D(256, 1, 1, 'same')
        self.conv_c5_1x1 = Conv2D(256, 1, 1, 'same')
        self.conv_c3_3x3 = Conv2D(256, 3, 1, 'same')
        self.conv_c4_3x3 = Conv2D(256, 3, 1, 'same')
        self.conv_c5_3x3 = Conv2D(256, 3, 1, 'same')
        self.conv_c6_3x3 = Conv2D(256, 3, 2, 'same')
        self.conv_c7_3x3 = Conv2D(256, 3, 2, 'same')
        self.upsample_2x = UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    head = Sequential([Input(shape=[None, None, 256])], name='images')
    rn_i = tf.initializers.RandomNormal(0.0, 0.01)

    for _ in range(4):
        head.add(Conv2D(256, 3, padding="same", kernel_initializer=rn_i))
        head.add(Activation('relu'))
    head.add(
        Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=rn_i,
            bias_initializer=bias_init,
        )
    )
    return head


class RetinaNet(Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)
