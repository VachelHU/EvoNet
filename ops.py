# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import json

"""parsing and configuration"""
def parse_args(filepath):
    f = open(filepath, encoding='utf-8')
    args = json.load(f)
    f.close()

    return args


def one_hot(Y, n_class):
    values = np.array(Y)
    ohY = np.eye(n_class, dtype=np.float32)[values]
    return ohY

def splitvalidate(datalength):
    shuffle_indices = np.random.permutation(np.arange(datalength))
    validateIndices = np.linspace(0, datalength, num=int(0.1 * datalength), endpoint=False).astype(np.int32)
    trainIndices = np.delete(np.arange(datalength), validateIndices, axis=0)
    return shuffle_indices, trainIndices, validateIndices

def cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], y_shapes[2]])], 2)


def conv1d(input_, output_dim, k_w=5, d=2, stddev=0.5, name="conv1d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, w, stride=d, padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)