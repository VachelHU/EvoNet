# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def one_hot(Y, n_class):
    values = np.array(Y)
    ohY = np.eye(n_class, dtype=np.float32)[values]
    return ohY

def splitvalidate(datalength):
    shuffle_indices = np.random.permutation(np.arange(datalength))
    validateIndices = np.linspace(0, datalength, num=int(0.1 * datalength), endpoint=False).astype(np.int32)
    trainIndices = np.delete(np.arange(datalength), validateIndices, axis=0)
    return shuffle_indices, trainIndices, validateIndices

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)