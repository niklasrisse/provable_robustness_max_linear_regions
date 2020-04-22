"""
Model definitions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np
import scipy.io as io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class MLP:
    def __init__(self, flag_train, hps):
        self.flag_train = flag_train
        self.hps = hps
        self.activation = tf.nn.relu

        W, b = [], []
        hps.n_hs_ext = [hps.n_in] + hps.n_hs + [hps.n_out]
        for i in range(len(hps.n_hs_ext) - 1):
            w_init = tf.truncated_normal([hps.n_hs_ext[i], hps.n_hs_ext[i + 1]], mean=0,
                                         stddev=tf.sqrt(2.0 / hps.n_hs_ext[i]), seed=hps.seed)
            W.append(tf.Variable(w_init, name='weight_' + str(i)))
            b_init = tf.zeros([1, hps.n_hs_ext[i + 1]])
            b.append(tf.Variable(b_init, name='bias_' + str(i)))
        self.W, self.b = W, b

    def net(self, x):
        x = tf.reshape(x, [-1, int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])])
        y, y_list = x, []
        for i in range(len(self.hps.n_hs)):
            y = y @ self.W[i] + self.b[i]
            y_list.append(y)
            y = tf.nn.relu(y)
        y_list.append(y @ self.W[-1] + self.b[-1])
        return y_list


class CNN:
    def __init__(self, flag_train, hps):
        self.flag_train = flag_train
        self.hps = hps
        self.activation = tf.nn.relu

    @staticmethod
    def weight_variable(name, shape, fc=False):
        """weight_variable generates a weight variable of a given shape. Uses He initialization."""
        if not fc:
            n_in = shape[0] * shape[1] * shape[2]
        else:
            n_in = shape[0]
        init = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / n_in))
        weights = tf.get_variable('weights_' + name, initializer=init)
        return weights

    @staticmethod
    def bias_variable(name, shape):
        """ Creates a bias variable of a given shape.
        """
        init = tf.constant(0.0, shape=shape)
        return tf.get_variable('biases_' + name, initializer=init)


class LeNetSmall(CNN):
    def __init__(self, flag_train, hps, dataset):
        """
        Difference: no shared biases, conv with stride instead of avg pool
        :param flag_train:
        :param hps:
        """
        super().__init__(flag_train, hps)
        self.n_filters = [16, 32]
        self.strides = [2, 2]
        self.padding = 'SAME'
        self.dataset = dataset
        n_fc_hidden = 100

        conv_sizes = [4, 4]
        if self.padding == 'VALID':
            h_last_conv = w_last_conv = math.ceil((math.ceil((hps.height - conv_sizes[0] + 1) / 2) - conv_sizes[1] + 1) / 2)
        elif self.padding == 'SAME':
            h_last_conv = w_last_conv = math.ceil(math.ceil(hps.height / 2) / 2)
        else:
            raise ValueError('wrong padding')
        W = [self.weight_variable('conv1', [conv_sizes[0], conv_sizes[0], self.hps.n_col, self.n_filters[0]]),
             self.weight_variable('conv2', [conv_sizes[1], conv_sizes[1], self.n_filters[0], self.n_filters[1]]),
             self.weight_variable('fc1', [h_last_conv * w_last_conv * self.n_filters[1], n_fc_hidden], fc=True),
             self.weight_variable('fc2', [n_fc_hidden, self.hps.n_out], fc=True)
             ]
        b = [
            self.bias_variable('conv1', [self.n_filters[0]]),
            self.bias_variable('conv2', [self.n_filters[1]]),
            self.bias_variable('fc1', [n_fc_hidden]),
            self.bias_variable('fc2', [self.hps.n_out])
        ]
        self.W = W
        self.b = b

    def net(self, x):

        if self.dataset == "mnist" or self.dataset == "fmnist":
            x = tf.reshape(x, (1,28,28,1))
        else:
            x = tf.reshape(x, (1,32,32,3))

        y_list = []
        y_list.append(x)
        for i in range(2):
            x = tf.nn.conv2d(x, self.W[i], strides=[1, self.strides[i], self.strides[i], 1], padding=self.padding) + self.b[i]  # bs x 12 x 12 x 16
            y_list.append(x)
            x = self.activation(x)
            y_list.append(x)

        x = tf.reshape(x, [-1, int(x.shape[1] * x.shape[2] * x.shape[3])])  # bs x 4*4*32
        x = x @ self.W[2] + self.b[2]
        y_list.append(x)
        x = self.activation(x)
        y_list.append(x)

        logits = x @ self.W[3] + self.b[3]
        y_list.append(logits)
        return y_list

def load_model(sess, args, model_path):
    
    _input = tf.placeholder(tf.float32, (1 , args.height, args.width, args.n_col))
    input_expanded = tf.expand_dims(_input, axis=0)
    
    if args.nn_type=="cnn":
        model = LeNetSmall(False, args, args.dataset)
    else:
        model = MLP(False, hps)

    _logits = model.net(input_expanded)[-1]
    _activations = model.net(input_expanded)

    param_file = io.loadmat(model_path)

    if args.nn_type=="cnn":
        weight_names = ['weights_conv1', 'weights_conv2', 'weights_fc1', 'weights_fc2']
        bias_names = ['biases_conv1', 'biases_conv2', 'biases_fc1', 'biases_fc2']
    else:
        weight_names = ['U', 'W']
        bias_names = ['bU', 'bW']

    for var_tf, var_name_mat in zip(model.W, weight_names):
        var_tf.load(param_file[var_name_mat], sess)
    for var_tf, var_name_mat in zip(model.b, bias_names):
        bias_val = param_file[var_name_mat]
        if args.nn_type=="cnn":
            bias_val = bias_val.flatten()
        var_tf.load(bias_val, sess)
            
    return model, _input, _logits, _activations