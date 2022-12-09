#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from itertools import chain
rng = np.random.default_rng()
import sys

pad_same = "SAME"
pad_valid = "VALID"

class Unit3D(tf.Module):
    #Basic unit containing Conv3D + BatchNorm + non-linearity.
    def __init__(self, output_channels,
                   kernel_shape = [1,1,1],
                   stride=[1, 1, 1],
                   activation_fn=tf.nn.relu,
                   use_batch_norm=True,
                   use_bias=False,
                   name='unit_3d'):
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._padding = pad_same
        self._stride = [1] + stride + [1]
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self._name = name

        # vardict refers to a global dictionary of tf.Variables loaded from the file containing the weights
        if self._use_batch_norm:
            self.bn_beta = vardict[self._name + "/batch_norm/beta"]
            self.bn_moving_mean = vardict[self._name + "/batch_norm/moving_mean"]
            self.bn_moving_variance = vardict[self._name + "/batch_norm/moving_variance"]
        self.conv_w = vardict[self._name + "/conv_3d/w"]
        if self._use_bias:
            self.conv_b = vardict[self._name+"/conv_3d/b"]
        
    def __call__(self, inputs, is_training):
         # input shape is [batch, depth, height, width, channels]
        net = tf.nn.conv3d(inputs, filters=self.conv_w, strides=self._stride, padding=self._padding)
        if self._use_bias:
            net = tf.nn.bias_add(net, self.conv_b)
        if self._use_batch_norm:
            net = tf.nn.batch_normalization(net, 
                                            self.bn_moving_mean, 
                                            self.bn_moving_variance, 
                                            self.bn_beta, 
                                            scale=1, 
                                            variance_epsilon=0.01)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class InceptionI3d(tf.Module):
    #  """Inception-v1 I3D architecture.
    #  The model is introduced in:
    #    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    #    Joao Carreira, Andrew Zisserman
    #    https://arxiv.org/pdf/1705.07750v1.pdf.
    #  See also the Inception architecture, introduced in:
    #    Going deeper with convolutions
    #    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    #    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    #    http://arxiv.org/pdf/1409.4842v1.pdf.
    #  """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions')
    
    # In the paper referenced above, notations are made of the receptive field after each pooling layer, i.e. the
    # size of the input data that each of its outputs depends on. They are (time is the first dim listed):
    #   MaxPool3d_2a_3x3   7x11x11
    #   MaxPool3d_3a_3x3   11x27x27
    #   MaxPool3d_4a_3x3   23x75x75
    #   MaxPool3d_5a_2x2   59x219x219
    #   AvgPool3d_2x7x7    99x539x539
    # The AvgPool layer is immediately prior to the logits.
    # Since the net uses only convolutional layers the dimensions of its input are not fixed (they can even vary
    # between calls, since the net calls tf.nn.conv3d directly rather than keras.layers.Conv3D).
    

    def __init__(self, var_prefix='RGB', num_classes=400, spatial_squeeze=True,
               final_endpoint='Logits', name='inception_i3d'):
        #    """Initializes I3D model instance.
        
        #    Args:
        #      num_classes: The number of outputs in the logit layer (default 400, which
        #          matches the Kinetics dataset).
        #      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
        #          before returning (default True).
        #      final_endpoint: The model contains many possible endpoints.
        #          `final_endpoint` specifies the last endpoint for the model to be built
        #          up to. In addition to the output at `final_endpoint`, all the outputs
        #          at endpoints up to `final_endpoint` will also be returned, in a
        #          dictionary. `final_endpoint` must be one of
        #          InceptionI3d.VALID_ENDPOINTS (default 'Logits').
        #      name: A string (optional). The name of this module.
        #    Raises:
        #      ValueError: if `final_endpoint` is not recognized.
        #    """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._var_prefix = var_prefix

        # except for the first and last entries here all this (output channels and kernel shape) is already implicit 
        # in the weights passed to the modules. the important part is the correspondence between modules and the 
        # names of variables contained in the checkpoint data
        arg_dict = {'Conv3d_1a_7x7' : {'output_channels': 64, 'kernel_shape' : [7,7,7], 'stride' : [2,2,2]}, 
                     'Conv3d_2b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Conv3d_2c_3x3' : {'output_channels' : 192, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_3b/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3b/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 96, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3b/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 128, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_3b/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 16, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3b/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 32, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_3b/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 32, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3c/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3c/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3c/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 192, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_3c/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 32, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_3c/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 96, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_3c/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4b/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 192, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4b/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 96, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4b/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 208, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4b/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 16, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4b/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 48, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4b/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4c/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 160, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4c/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 112, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4c/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 224, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4c/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 24, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4c/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 64, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4c/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4d/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4d/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4d/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 256, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4d/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 24, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4d/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 64, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4d/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4e/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 112, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4e/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 144, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4e/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 288, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4e/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 32, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4e/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 64, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4e/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 64, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4f/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 256, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4f/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 160, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4f/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 320, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4f/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 32, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_4f/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 128, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_4f/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5b/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 256, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5b/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 160, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5b/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 320, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_5b/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 32, 'kernel_shape' : [1, 1, 1]},
                    # typo here: in other modules the name is Branch2/Conv3d_0b_3x3 !
                     'Mixed_5b/Branch_2/Conv3d_0a_3x3' : {'output_channels' : 128, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_5b/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5c/Branch_0/Conv3d_0a_1x1' : {'output_channels' : 384, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5c/Branch_1/Conv3d_0a_1x1' : {'output_channels' : 192, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5c/Branch_1/Conv3d_0b_3x3' : {'output_channels' : 384, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_5c/Branch_2/Conv3d_0a_1x1' : {'output_channels' : 48, 'kernel_shape' : [1, 1, 1]},
                     'Mixed_5c/Branch_2/Conv3d_0b_3x3' : {'output_channels' : 128, 'kernel_shape' : [3, 3, 3]},
                     'Mixed_5c/Branch_3/Conv3d_0b_1x1' : {'output_channels' : 128, 'kernel_shape' : [1, 1, 1]},
                     'Logits/Conv3d_0c_1x1' : {'output_channels' : self._num_classes, 'kernel_shape' : [1, 1, 1],
                                              'activation_fn' : None, 'use_batch_norm' : False, 'use_bias' : True}
                    }
        
        self.module_dict = {}
        model_prefix = self._var_prefix + "/" + self.name + "/" 
        for module_name in list(arg_dict.keys()):
            self.module_dict[module_name] = Unit3D(**arg_dict[module_name],
                                              name = model_prefix+module_name)

    def __call__(self, inputs, is_training = False, dropout_prob=0.0):
        #    """Connects the model to inputs.
        #    Args:
        #      inputs: Inputs to the model, which should have dimensions
        #          `step_size` x `num_frames` x 224 x 224 x `num_channels`.
        #      is_training: whether to use training mode for snt.BatchNorm (boolean).
        #      dropout_prob: Probability for the tf.nn.dropout layer (float in
        #          [0, 1)).
        #    Returns:
        #      A tuple consisting of:
        #        1. Network output at location `self._final_endpoint`.
        #        2. Dictionary containing all endpoints up to `self._final_endpoint`,
        #           indexed by endpoint name.
        #    Raises:
        #      ValueError: if `self._final_endpoint` is not recognized.
        #    """
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv3d_1a_7x7'        
        net = self.module_dict[end_point](net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        
        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=pad_same, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2b_1x1'
        net = self.module_dict[end_point](net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2c_3x3'
        net = self.module_dict[end_point](net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=pad_same, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_4a_3x3'
        # modified: was time stride = 2
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=pad_same, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                               padding=pad_same, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        # typo here: in other modules the name is Branch2/Conv3d_0b_3x3 !
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        branch_0 = self.module_dict[end_point+'/Branch_0/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_1 = self.module_dict[end_point+'/Branch_1/Conv3d_0b_3x3'](branch_1, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0a_1x1'](net, is_training=is_training)
        branch_2 = self.module_dict[end_point+'/Branch_2/Conv3d_0b_3x3'](branch_2, is_training=is_training)
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=pad_same)
        branch_3 = self.module_dict[end_point+'/Branch_3/Conv3d_0b_1x1'](branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                                 strides=[1, 1, 1, 1, 1], padding=pad_valid)
        net = tf.nn.dropout(net, dropout_prob)
        logits = self.module_dict[end_point+'/Conv3d_0c_1x1'](net, is_training=is_training)
        if self._spatial_squeeze:
            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points[end_point] = averaged_logits
        if self._final_endpoint == end_point: return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions, end_points

def ready_i3d_net(fine_tuning=False):
    # paths to pre-trained i3d models provided by the paper's authors
    _CHECKPOINT_PATHS = {'rgb'          : 'i3d/data/checkpoints/rgb_scratch/model.ckpt',
                         'rgb600'       : 'i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
                         'flow'         : 'i3d/data/checkpoints/flow_scratch/model.ckpt',
                         'rgb_imagenet' : 'i3d/data/checkpoints/rgb_imagenet/model.ckpt',
                         'flow_imagenet': 'i3d/data/checkpoints/flow_imagenet/model.ckpt',
    }

    # this list has the names of all the weights in the Flow net and their shapes
    flow_varlist = tf.train.list_variables(_CHECKPOINT_PATHS['flow_imagenet'])
    flow_vardict = {}
    # make variables to load the saved weights into
    for variable in flow_varlist:
        flow_vardict[variable[0]] = tf.Variable(initial_value = np.zeros(variable[1], dtype=np.float32),
                                                shape=tf.TensorShape(variable[1]),
                                                trainable=fine_tuning,
                                                name=variable[0])

    flow_saver = tf.compat.v1.train.Saver(var_list=flow_vardict)
    flow_saver.restore(sess=None, save_path=_CHECKPOINT_PATHS['flow_imagenet'])

    rgb_varlist = tf.train.list_variables(_CHECKPOINT_PATHS['rgb_imagenet'])
    rgb_vardict = {}
    for variable in rgb_varlist:
        rgb_vardict[variable[0]] = tf.Variable(initial_value = np.zeros(variable[1], dtype=np.float32),
                                               shape=tf.TensorShape(variable[1]), 
                                               trainable=fine_tuning,
                                               name=variable[0])
    rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_vardict, reshape=True)
    rgb_saver.restore(sess=None, save_path=_CHECKPOINT_PATHS['rgb_imagenet'])

    # now vardict will contain all the weights
    global vardict
    vardict = {}
    vardict.update(rgb_vardict)
    vardict.update(flow_vardict)

ready_i3d_net()

l2_coef = 1/2048
dropout_prob = 0.5
spatial_dropout_prob=0.02

learning_rate=0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    learning_rate,
                    decay_steps=500,
                    decay_rate=0.9,
                    staircase=True)

#opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
opt=tf.keras.optimizers.Nadam(learning_rate=0.0001) # but Nadam doesn't support schedules
# Nadam, adam with Nesterov momentum!

# two functions to make synthetic data
# however, it's possible tf's rng is contributing to some errors, and we're not really training long enough
# to take advantage of it, so we'll take it out
pseudo_random_bits = tf.queue.FIFOQueue(256,tf.bool)
pseudo_random_bits.enqueue_many(tf.cast([1,0,0,0,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,
                                         1,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,
                                         0,0,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,1,
                                         1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,
                                         0,1,0,0,0,1,0,1,0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,1,1,
                                         0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,1,
                                         1,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0], tf.bool))
def pseudo_random_bit():
    a_bit = pseudo_random_bits.dequeue()
    pseudo_random_bits.enqueue(a_bit)
    return a_bit    

def random_flip(seq):
    #if tf.cast(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1), tf.bool):
    if pseudo_random_bit():
        return tf.raw_ops.Reverse(tensor=seq, dims=[False,False,False,True,False])
    return seq

def random_augment(seq):
    seq = random_flip(seq)
    seq = tf.image.random_brightness(seq, 0.15)
    seq = tf.image.random_saturation(seq, 0.85, 1.15)
    seq = tf.image.random_contrast(seq, 0.85, 1.15)
    return tf.raw_ops.ClipByValue(t=seq, clip_value_min=0, clip_value_max=1)

# some pooling functions to reduce spatial dimensions output from convolutional layers
pool_args = {'ksize' : [1,1,2,2,1], 'strides' : [1,1,2,2,1], 'padding' : pad_same}

def pool_avg_or_max(filter_tensor, parity):
    return (tf.cond(parity,
                    lambda: tf.nn.avg_pool3d(filter_tensor, **pool_args),
                    lambda: tf.nn.max_pool3d(filter_tensor, **pool_args)),
            tf.logical_not(parity))

def pool_to_size(filter_tensor, target_size):
    in_shape = filter_tensor.shape
    parity = True
    return tf.while_loop(lambda t, p: tf.math.less(target_size, tf.shape(t)[-2]),
                 pool_avg_or_max,
                  (filter_tensor, parity),
                  shape_invariants=(tf.TensorShape([step_size, in_shape[1], None, None, in_shape[-1]]),
                                    tf.TensorShape(())))[0]

def pool_dict(tensor_dict, keys, sizes_dict=None, uniform_size=None):
    # if an integer is passed for sizes, pool everything down to that size
    if uniform_size is not None:
        for key in keys:
            tensor_dict.update({key : pool_to_size(tensor_dict[key], uniform_size)})
            tf.ensure_shape(tensor_dict[key], (step_size, None, uniform_size, uniform_size, None))
            
    # otherwise, pool to the specified sizes
    elif sizes_dict is not None:
        for key in keys:
            tensor_dict.update({key : pool_to_size(tensor_dict[key], sizes_dict[key])})
            tf.ensure_shape(tensor_dict[key], (step_size, None, sizes_dict[key], sizes_dict[key],None))
    return 0

# applied to model weights, this is L^2 regularisation
def var_list_decay(var_list, coef):
    for j in range(len(var_list)):
        var_list[j].assign_sub(var_list[j]*coef)

# some informational statistics for the logfiles
def get_var_list_stats(var_list):
    norm_acc, number_acc = 0, 0
    for j in range(len(var_list)):
        number_acc += tf.math.reduce_prod(var_list[j].shape)
        norm_acc += tf.math.reduce_euclidean_norm(var_list[j])
    return norm_acc.numpy(), number_acc.numpy(), norm_acc.numpy()/number_acc.numpy()

def write_var_list_stats(var_list, name, writer, step):
    with writer.as_default():
        for j in range(len(var_list)):
            tf.summary.histogram(name = name+"/"+str(j), 
                                 data = var_list[j],
                                 step = step)
    return 0

def write_loss_stat(loss_, writer, step):
    with writer.as_default():
        tf.summary.scalar(name='loss',
                         data = loss_,
                         step = step)


# model configuration constants

step_size = 1 

shuffle_buffer_size = None # only used for shuffling the dataset

alphabet_size = 32 # 26 letters, 5 punctuation symbols, and the 'blank' symbol

c2d_filters=(64,64,128,128)
attender_units = 64
cogitator_units = 256
decoder_units = 128

i3d_endpoints = ['Mixed_4c', 'MaxPool3d_4a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_2a_3x3']
#i3d_endpoints = ['MaxPool3d_4a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_2a_3x3']
encoder_endpoints = ['Mixed_4c', 'MaxPool3d_4a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_2a_3x3', 'conv_2d']
target_sizes = {'Mixed_4c' : 4, 'MaxPool3d_4a_3x3' : 4, 'MaxPool3d_3a_3x3' : 8, 
                         'MaxPool3d_2a_3x3' : 8, 'conv_2d' : 8}
encoder_filters = {'Mixed_4c' : 512, 'MaxPool3d_4a_3x3' : 480, 'MaxPool3d_3a_3x3' : 192, 
                   'MaxPool3d_2a_3x3' : 64, 'conv_2d' : c2d_filters[-1]}
projection_dims = {key: 128 for key in encoder_endpoints}
projection_dim_total = 2*128*len(encoder_endpoints)
encoder_target_outputs = {key: encoder_filters[key]*target_sizes[key]*target_sizes[key] for key in encoder_endpoints}
encoder_pooled_output = np.array([encoder_filters[key]*target_sizes[key]*target_sizes[key] 
                                  for key in encoder_endpoints]).sum()
                         
filter_count = np.sum([encoder_filters[endpt] for endpt in encoder_endpoints])
decoder_in_size = 2*filter_count+cogitator_units


# In[6]:


@tf.custom_gradient
def softsign_shift(x):
    def grad(upstream):
        return upstream / tf.math.pow(1 + tf.math.abs(x), 2)
    return tf.nn.softsign(x) + 0.5, grad

def dense_generate_weights(dense_layer, in_size):
    dense_layer(tf.random.uniform((1,1,in_size)))

def dense_save_weights(dense_layer, out_dir, name):
    tf.io.write_file(out_dir + name + "_dense_w", tf.io.serialize_tensor(dense_layer.variables[0]))
    tf.io.write_file(out_dir + name + "_dense_b", tf.io.serialize_tensor(dense_layer.variables[1]))

def dense_load_weights(dense_layer, in_dir, name, in_size=None):
    if len(dense_layer.variables) == 0 and in_size is not None:
        dense_generate_weights(dense_layer, in_size)
    dense_layer.variables[0].assign(tf.io.parse_tensor(tf.io.read_file(in_dir+name+"_dense_w"), out_type = tf.float32))
    dense_layer.variables[1].assign(tf.io.parse_tensor(tf.io.read_file(in_dir+name+"_dense_b"), out_type = tf.float32))
    
def lstm_save_weights(lstm_layer, out_dir, name):
    tf.io.write_file(out_dir + name + "_lstm_w", tf.io.serialize_tensor(lstm_layer.variables[0]))
    tf.io.write_file(out_dir + name + "_lstm_rw", tf.io.serialize_tensor(lstm_layer.variables[1]))
    tf.io.write_file(out_dir + name + "_lstm_b", tf.io.serialize_tensor(lstm_layer.variables[2]))
    
def lstm_load_weights(lstm_layer, in_dir, name):
    lstm_layer.variables[0].assign(tf.io.parse_tensor(tf.io.read_file(in_dir+name + "_lstm_w"), out_type=tf.float32))
    lstm_layer.variables[1].assign(tf.io.parse_tensor(tf.io.read_file(in_dir+name + "_lstm_rw"), out_type=tf.float32))
    lstm_layer.variables[2].assign(tf.io.parse_tensor(tf.io.read_file(in_dir+name + "_lstm_b"), out_type=tf.float32))

# calls the i3d model on RGB and flow frame sequences
class testSpeller3dEncoder(tf.Module):
    def __init__(self, i3d_endpoint='MaxPool3d_4a_3x3', fine_tuning=False):
        super(testSpeller3dEncoder, self).__init__()
        self.fine_tuning = fine_tuning
        self.rgb = InceptionI3d(var_prefix='RGB', final_endpoint=i3d_endpoint)
        self.flow = InceptionI3d(var_prefix='Flow',final_endpoint=i3d_endpoint)
        
    def __call__(self, inputs, is_training=False, return_dict=True):
        # input shape [batch, time, height, width, channels]
        #rgb_scaled = random_augment(tf.image.convert_image_dtype(inputs[0].to_tensor(), dtype=tf.float32))
        rgb_scaled = random_flip(tf.image.convert_image_dtype(inputs[0].to_tensor(), dtype=tf.float32))
        rgb_results = self.rgb(rgb_scaled, self.fine_tuning)
        # i3d expects flow in range [-1,1]
        flow_results = self.flow(2*random_flip(tf.image.convert_image_dtype(inputs[1].to_tensor(), dtype=tf.float32)-1),
                                 self.fine_tuning)
        if return_dict:
            return rgb_results[1], flow_results[1]
        return rgb_results[0], flow_results[0]
    
class testSpellerAttender(tf.Module):
    def __init__(self, units=64, attendee_size = encoder_pooled_output):
        self.attendee_size = attendee_size
        self.geom1 = tf.keras.layers.Dense(units, activation='relu')
        self.geom2 = tf.keras.layers.Dense(units, activation='relu')
        
        self.module_dicts = [{key: tf.keras.layers.Dense(encoder_target_outputs[key], activation='softmax')
                              for key in encoder_endpoints} for j in [0,1]]
        self.pool_args = {'ksize' : [1,2,1], 'strides': [1,2,1], 'padding' : pad_same}
        
    def __call__(self, geom_in, is_training=True):
        # pool w/ stride 2 in time dim to agree with depth reduction in 3dEncoder
        out = self.geom2(self.geom1(geom_in.to_tensor(shape=(step_size,None,6*21))))
        results_dicts=[{key : tf.reshape(tf.nn.avg_pool(self.module_dicts[j][key](out), **self.pool_args),
                                            (step_size, -1, target_sizes[key], target_sizes[key], encoder_filters[key]))
                              for key in encoder_endpoints}
                       for j in [0,1]]
        return results_dicts
        
    def generate_weights(self):
        self.__call__(tf.RaggedTensor.from_tensor(tf.random.uniform((1,1,6*21))))
        
    def save_weights(self, out_dir):
        dense_save_weights(self.geom1, out_dir, 'geom1')
        dense_save_weights(self.geom2, out_dir, 'geom2')
        for j in [0,1]:
            for key, mod in self.module_dicts[j].items():
                dense_save_weights(mod, out_dir, 'attend_'+str(j)+"_"+key)
    
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights()
        dense_load_weights(self.geom1, in_dir, 'geom1')
        dense_load_weights(self.geom2, in_dir, 'geom2')
        for j in [0,1]:
            for key, mod in self.module_dicts[j].items():
                dense_load_weights(mod, in_dir, 'attend_'+str(j)+"_"+key)
        
class testSpellerProjector(tf.Module):
    def __init__(self, units_dict={}, units=128, dropout_prob=0):
        super(testSpellerProjector, self).__init__()
        self.module_dict = {}
        for key in encoder_endpoints:
            self.module_dict[key] = [tf.keras.layers.Dense(units_dict.get(key, units), activation=None),
                                     tf.keras.layers.Dense(units_dict.get(key, units), activation=None)]
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
    
    def __call__(self, input_dicts, results_depth, is_training=True):
        results_dict = {}
        for key in encoder_endpoints:
            results_dict[key] = tf.concat([self.dropout(
                                                self.module_dict[key][0](
                                                    tf.reshape(input_dicts[0][key],
                                                               (step_size, results_depth, encoder_target_outputs[key]))),
                                                training=is_training),
                                           self.dropout(
                                               self.module_dict[key][1](
                                                   tf.reshape(input_dicts[0][key], 
                                                              (step_size, results_depth, encoder_target_outputs[key]))),
                                                training=is_training)],
                                          axis=-1)
        return results_dict
    
    def generate_weights(self, in_size_dict=encoder_target_outputs):
        for key, mod in self.module_dict.items():
            for j in [0,1]:
                dense_generate_weights(mod[j], in_size_dict[key])
    
    def save_weights(self, out_dir):
        for key, mod in self.module_dict.items():
            for j in [0,1]:
                dense_save_weights(mod[j], out_dir, 'proj_' +str(j) + '_' + key)
            
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights()
        for key, mod in self.module_dict.items():
            for j in [0,1]:
                dense_load_weights(mod[j], in_dir, 'proj_' +str(j) + '_' + key)
        
class testSpellerCogitator(tf.Module):
    def __init__(self, units=256):
        super(testSpellerCogitator, self).__init__()
        # i/o shape [batch, timesteps, channels] with return_sequences on
        # out shape [batch, units] with return_sequences off
        self.insize = projection_dim_total
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True, recurrent_activation=softsign_shift)
        self.lstm2 = tf.keras.layers.LSTM(units, return_sequences=True, recurrent_activation=softsign_shift)
        self.lstm3 = tf.keras.layers.LSTM(units, return_sequences=True, recurrent_activation=softsign_shift)
    
    def __call__(self, inputs, is_training=True):
        return self.lstm3(self.lstm2(self.lstm1(inputs)))
    
    def generate_weights(self, in_size=projection_dim_total):
        self.__call__(tf.random.uniform((1,1,in_size)))
    
    def save_weights(self, out_dir):
        lstm_save_weights(self.lstm1, out_dir, 'lstm1')
        lstm_save_weights(self.lstm2, out_dir, 'lstm2')
        lstm_save_weights(self.lstm3, out_dir, 'lstm3')
        
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights()
        lstm_load_weights(self.lstm1, in_dir, 'lstm1')
        lstm_load_weights(self.lstm2, in_dir, 'lstm2')
        lstm_load_weights(self.lstm3, in_dir, 'lstm3')  
    
class testSpellerDecoder(tf.Module):
    def __init__(self, units, dropout_prob, labels=32):
        super(testSpellerDecoder,self).__init__()
        self.in_size = decoder_in_size
        self.class1 = tf.keras.layers.Dense(units, activation = 'relu')
        self.class2 = tf.keras.layers.Dense(units, activation = 'relu')
        self.class3 = tf.keras.layers.Dense(labels)
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        
    def __call__(self, input_, is_training=True):
        out = self.dropout(self.class1(tf.raw_ops.EnsureShape(input = input_,
                                                              shape = (step_size, None, decoder_in_size))), training=is_training)
        out = self.dropout(self.class2(out), training = is_training)
        return self.class3(out)
    
    def generate_weights(self, in_size=decoder_in_size):
        self.__call__(tf.random.uniform((step_size, 1, in_size)))
    
    def save_weights(self, out_dir):
        dense_save_weights(self.class1, out_dir, 'class1')
        dense_save_weights(self.class2, out_dir, 'class2')
        dense_save_weights(self.class3, out_dir, 'class3')
    
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights(self.in_size)
        dense_load_weights(self.class1, in_dir, 'class1')
        dense_load_weights(self.class2, in_dir, 'class2')
        dense_load_weights(self.class3, in_dir, 'class3')
        
# following the idea of presenting the LSTM middle of the network with information at variable time-scales,
# we consider also a 2d convolutional network that operates only on individual frames

# a 2d convolutional layer padded with a depth axis of length 1
class Unit2d(tf.Module):
    def __init__(self, input_channels,
                 output_channels,
                 rec_field_length = 3,
                 activation=tf.nn.relu,
                 use_bias=True,
                 padding=pad_same,
                 name='unit2d'):
        super(Unit2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rec_field_length = rec_field_length
        self._name = name
        self.conv_w = tf.Variable(initial_value = tf.keras.initializers.glorot_uniform()((1, rec_field_length, 
                                                                                          rec_field_length, 
                                                                                          input_channels, 
                                                                                          output_channels)),
                                  trainable=True, name=name+"/w")
        self.use_bias = use_bias
        if self.use_bias:
            self.conv_b = tf.Variable(initial_value = tf.keras.initializers.zeros()((output_channels)),
                                      trainable=True, name=name+"/b")
        self.strides = [1,1,1,1,1]
        self.padding = padding
        self.activation = activation

    def __call__(self, inputs):
        out = tf.nn.conv3d(inputs, filters=self.conv_w, strides=self.strides, padding=self.padding)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.conv_b)
        if self.activation is not None:
            out = self.activation(out)
        return out
    
    def generate_weights(self):
        self.__call__(tf.random.uniform((1,1,1,1,self.input_channels)))
        
    def save_weights(self, out_dir):
        # this gives a utf-8 encoding error
        #tf.io.write_file(tf.io.serialize_tensor(self.conv_w), out_dir+self._name+"_conv_w")
        #tf.io.write_file(tf.io.serialize_tensor(self.conv_b), out_dir+self._name+"_conv_b")
        np.save(out_dir+self._name+"_conv_w.npy", self.conv_w)
        np.save(out_dir+self._name+"_conv_b.npy", self.conv_b)
    
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights()
        #self.conv_w.assign(tf.io.parse_tensor(tf.io.read_file(in_dir+self._name+"_conv_w.npy"), out_type=tf.float32))
        #self.conv_b.assign(tf.io.parse_tensor(tf.io.read_file(in_dir+self._name+"_conv_b.npy"), out_type=tf.float32))
        self.conv_w.assign(np.load(in_dir+self._name+"_conv_w.npy"))
        self.conv_b.assign(np.load(in_dir+self._name+"_conv_b.npy"))
        

class testSpeller2dEncoder(tf.Module):
    def __init__(self, conv_filters, spatial_dropout_prob=0.02):
        super(testSpeller2dEncoder, self).__init__()
        filters_1, filters_2, filters_3, filters_4 = conv_filters
        self.spatial_dropout_prob = spatial_dropout_prob
        
        # input channels, output channels, rec field length 
        self.conv_1a_rgb = Unit2d(3, filters_1, 5, name='conv_1a_rgb')
        self.conv_1b_rgb = Unit2d(filters_1, filters_1, name='conv_1b_rgb')
        self.conv_2a_rgb = Unit2d(filters_1, filters_2, name='conv_2a_rgb')
        self.conv_2b_rgb = Unit2d(filters_2, filters_2, name='conv_2b_rgb')
        self.conv_3a_rgb = Unit2d(filters_2, filters_3, name='conv_3a_rgb')
        self.conv_3b_rgb = Unit2d(filters_3, filters_3, name='conv_3b_rgb')
        self.conv_4a_rgb = Unit2d(filters_3, filters_4, name='conv_4a_rgb')
        self.conv_4b_rgb = Unit2d(filters_4, filters_4, name='conv_4b_rgb')
        
        self.conv_1a_flow = Unit2d(2, filters_1, 5, name='conv_1a_flow')
        self.conv_1b_flow = Unit2d(filters_1, filters_1, name='conv_1b_flow')
        self.conv_2a_flow = Unit2d(filters_1, filters_2, name='conv_2a_flow')
        self.conv_2b_flow = Unit2d(filters_2, filters_2, name='conv_2b_flow')
        self.conv_3a_flow = Unit2d(filters_2, filters_3, name='conv_3a_flow')
        self.conv_3b_flow = Unit2d(filters_3, filters_3, name='conv_3b_flow')
        self.conv_4a_flow = Unit2d(filters_3, filters_4, name='conv_4a_flow')
        self.conv_4b_flow = Unit2d(filters_4, filters_4, name='conv_4b_flow')
        
        self.dropout = tf.keras.layers.SpatialDropout3D(self.spatial_dropout_prob)
        
        self.module_dict = {self.conv_1a_rgb.name : self.conv_1a_rgb, self.conv_1b_rgb.name : self.conv_1b_rgb,
                            self.conv_2a_rgb.name : self.conv_2a_rgb, self.conv_2b_rgb.name : self.conv_2b_rgb,
                            self.conv_3a_rgb.name : self.conv_3a_rgb, self.conv_3b_rgb.name : self.conv_3b_rgb,
                            self.conv_4a_rgb.name : self.conv_4a_rgb, self.conv_4b_rgb.name : self.conv_4b_rgb,
                            self.conv_1a_flow.name : self.conv_1a_flow, self.conv_1b_flow.name : self.conv_1b_flow,
                            self.conv_2a_flow.name : self.conv_2a_flow, self.conv_2b_flow.name : self.conv_2b_flow,
                            self.conv_3a_flow.name : self.conv_3a_flow, self.conv_3b_flow.name : self.conv_3b_flow,
                            self.conv_4a_flow.name : self.conv_4a_flow, self.conv_4b_flow.name : self.conv_4b_flow}

    def __call__(self, inputs, is_training=True):
        # 128x128x3 input
        out = tf.keras.layers.GaussianNoise(0.03)(tf.image.convert_image_dtype(inputs[0].to_tensor(), dtype=tf.float32),
                                                  training=is_training)
        out = self.dropout(self.conv_1a_rgb(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_1b_rgb(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)
        out = self.dropout(self.conv_2a_rgb(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_2b_rgb(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)
        out = self.dropout(self.conv_3a_rgb(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_3b_rgb(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)  
        out = self.dropout(self.conv_4a_rgb(out), training=is_training)
        rgb_out = tf.nn.max_pool3d(self.conv_4b_rgb(out),
                                   ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)
        
        # 128x128x2 input
        out = tf.image.convert_image_dtype(inputs[1].to_tensor(), dtype=tf.float32)
        out = self.dropout(self.conv_1a_flow(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_1b_flow(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)
        out = self.dropout(self.conv_2a_flow(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_2b_flow(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)
        out = self.dropout(self.conv_3a_flow(out), training=is_training)
        out = tf.nn.max_pool3d(self.conv_3b_flow(out),
                               ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)  
        out = self.dropout(self.conv_4a_flow(out), training=is_training)
        flow_out = tf.nn.max_pool3d(self.conv_4b_flow(out),
                                   ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding=pad_same)

        # pool w/ stride 2 in time dim to agree with depth reduction in 3dEncoder
        # (could also change the time stride on final max pool if preferred)
        return tf.nn.avg_pool3d(rgb_out, ksize=[1,2,1,1,1], strides=[1,2,1,1,1], padding=pad_same), tf.nn.avg_pool3d(flow_out, ksize=[1,2,1,1,1], strides=[1,2,1,1,1], padding=pad_same)
    
    def generate_weights(self):
        for module in self.module_dict.values():
            module.generate_weights()
            
    def save_weights(self, out_dir):
        for module in self.module_dict.values():
            module.save_weights(out_dir)
    
    def load_weights(self, in_dir):
        if len(self.variables) == 0:
            self.generate_weights()
        for module in self.module_dict.values():
            module.load_weights(in_dir)
            
def save_network(out_dir):
    testSpellAttend.save_weights(out_dir + 'attend/')
    tf.io.gfile.mkdir(out_dir+'encode/')
    testSpell2dEncode.save_weights(out_dir + 'encode/')
    testSpellProject.save_weights(out_dir + 'proj/')
    testSpellCogitate.save_weights(out_dir + 'cog/')
    testSpellDecode.save_weights(out_dir + 'decode/')

def load_network(in_dir):
    global testSpellAttend, testSpell2dEncode, testSpellProject, testSpellCogitate, testSpellDecode
    testSpellAttend.load_weights(in_dir + 'attend/')
    testSpell2dEncode.load_weights(in_dir + 'encode/')
    testSpellProject.load_weights(in_dir  + 'proj/')
    testSpellCogitate.load_weights(in_dir + 'cog/')
    testSpellDecode.load_weights(in_dir + 'decode/')
    
def make_network():
    global testSpellAttend, testSpell2dEncode, testSpellProject, testSpellCogitate, testSpellDecode, testSpell3dEncode
    testSpell2dEncode = testSpeller2dEncoder((64,64,128,128), spatial_dropout_prob=spatial_dropout_prob)
    testSpell3dEncode = testSpeller3dEncoder(i3d_endpoint = i3d_endpoints[0])
    testSpellAttend = testSpellerAttender(attender_units, encoder_pooled_output)
    testSpellProject = testSpellerProjector()
    testSpellCogitate = testSpellerCogitator(units = cogitator_units)
    testSpellDecode = testSpellerDecoder(units = decoder_units, dropout_prob = dropout_prob, labels = alphabet_size)

def generate_network_weights():
    global testSpellAttend, testSpell2dEncode, testSpellProject, testSpellCogitate, testSpellDecode
    testSpell2dEncode.generate_weights()
    testSpellAttend.generate_weights()
    testSpellProject.generate_weights()
    testSpellCogitate.generate_weights()
    testSpellDecode.generate_weights()
    
def reboot_network(save_dir):
    global testSpellAttend, testSpell2dEncode, testSpellProject, testSpellCogitate, testSpellDecode, testSpell3dEncode
    save_network(save_dir)
    del testSpellAttend, testSpell2dEncode, testSpellProject, testSpellCogitate, testSpellDecode, testSpell3dEncode
    make_network()
    load_network(save_dir)

make_network()

save_path = './spell_net/'
tf.io.gfile.mkdir(save_path)

base_cfsw  = sys.argv[1]#"h:/project/cfsw/"
base_cfswp = sys.argv[2]#"h:/project/cfswp/"

cfsw_shards, cfsw_shard_size = 21, 256
cfswp_shards, cfswp_shard_size = 24, 512 ## 48
total_shards = cfsw_shards + cfswp_shards
total_examples = cfsw_shards*cfsw_shard_size+cfswp_shards*cfswp_shard_size

def load_fs_shards(start_shard, endpoint, 
                   image_in, geom_in, label_in, image_tag,
                   step_size = step_size, image_size=128, file_size=256, buffer_size=step_size,
                   return_tf_datasets = True, use_shuffle=True):
    for shard_number in range(start_shard, endpoint):
        suffix = str(shard_number) + ".npy"
        image_suffix = str(shard_number) + image_tag + ".npy"
        
        rgb_a = np.load(image_in + "train_img_a_" + image_suffix)
        flow_a = np.load(image_in + "train_flow_a_" + image_suffix)
        geom_a = np.concatenate([np.load(geom_in + "train_geom_img_a_" + suffix), # 21 * 3 floats per frame per file
                                 np.load(geom_in + "train_geom_wrl_a_" + suffix)], axis=1)
        rgb_b = np.load(image_in + "train_img_b_" + image_suffix)
        flow_b = np.load(image_in + "train_flow_b_" + image_suffix)
        geom_b = np.concatenate([np.load(geom_in + "train_geom_img_b_" + suffix),
                                 np.load(geom_in + "train_geom_wrl_b_" + suffix)], axis=1)
        label_seqs = np.int32(np.load(label_in + "train_label_seqs_" + suffix))
        label_lengths = np.load(label_in + "train_seq_lengths_" + suffix)
        frame_counts = np.load(label_in + "train_frame_counts_" + suffix)
        
        if not return_tf_datasets:
            return rgb_a, flow_a, geom_a, rgb_b, flow_b, geom_b, label_seqs, label_lengths, frame_counts

        seq_count = frame_counts.shape[0]
        seq_heights=[image_size]*file_size
        seq_widths = [image_size]*file_size
        # for input that hasn't been resized to a fixed section, these lines compute dimensions
        #seq_heights = [crnrs[1][1] - crnrs[0][1] for crnrs in corners[0]]
        #seq_widths = [crnrs[1][0] - crnrs[0][0] for crnrs in corners[0]]

        total_frames = frame_counts.sum()
        frame_heights = list(chain(*[[seq_heights[j]]*frame_counts[j] for j in range(seq_count)]))
        frame_widths = list(chain(*[[seq_widths[j]]*frame_counts[j] for j in range(seq_count)]))
        by_widths = list(chain(*[[frame_widths[j]]*frame_heights[j] for j in range(total_frames)]))
        
        # in the general case, the image tensors have shape 
        #    [shard_size, (frame_counts), (frame_heights), (frame_widths), channels]
        #  with the middle 3 dimensions ragged, and are constructed by 
        #
        #rgb_a_tensor = tf.RaggedTensor.from_row_lengths(tf.RaggedTensor.from_row_lengths(tf.RaggedTensor.from_row_lengths(rgb_a, 
        #                                                                                                              by_widths),
        #                                                                                frame_heights),
        #                                                frame_counts)
        #
        # Here we have fixed heights and widths, so we have a simpler construction of ragged rank 1.
        rgb_a_tensor = tf.RaggedTensor.from_row_lengths(
                            tf.RaggedTensor.from_uniform_row_length(
                                tf.RaggedTensor.from_uniform_row_length(rgb_a, 128),128),
                                                        frame_counts)
        del rgb_a 

        flow_a_tensor = tf.RaggedTensor.from_row_lengths(
                             tf.RaggedTensor.from_uniform_row_length(
                                tf.RaggedTensor.from_uniform_row_length(flow_a, 128),128),
                                                        frame_counts)
        del flow_a

        geom_a_tensor = tf.RaggedTensor.from_row_lengths(
                            tf.RaggedTensor.from_uniform_row_length(geom_a, 21), frame_counts).merge_dims(-2,-1)
        del geom_a

        rgb_b_tensor = tf.RaggedTensor.from_row_lengths(
                            tf.RaggedTensor.from_uniform_row_length(
                                tf.RaggedTensor.from_uniform_row_length(rgb_b, 128),128),
                                                        frame_counts)
        del rgb_b

        flow_b_tensor = tf.RaggedTensor.from_row_lengths(
                             tf.RaggedTensor.from_uniform_row_length(
                                tf.RaggedTensor.from_uniform_row_length(flow_b, 128),128),
                                                        frame_counts)
        del flow_b
 
        geom_b_tensor = tf.RaggedTensor.from_row_lengths(
                            tf.RaggedTensor.from_uniform_row_length(geom_b, 21), frame_counts).merge_dims(-2,-1)
        del geom_b

        # bundle the data into a tf Dataset with set step size
        label_seqs = tf.sparse.from_dense(label_seqs)
        if shard_number==start_shard:
            dataset_ = tf.data.Dataset.from_tensor_slices((((rgb_a_tensor, flow_a_tensor, geom_a_tensor), 
                                                            (rgb_b_tensor, flow_b_tensor, geom_b_tensor)), 
                                                           (label_seqs,label_lengths)))
        else:
            dataset_ = dataset_.concatenate(tf.data.Dataset.from_tensor_slices((((rgb_a_tensor, flow_a_tensor, geom_a_tensor), 
                                                                                 (rgb_b_tensor, flow_b_tensor, geom_b_tensor)), 
                                                                                (label_seqs, label_lengths))))
        del rgb_a_tensor, flow_a_tensor, geom_a_tensor
        del rgb_b_tensor, flow_b_tensor, geom_b_tensor
    
    if not use_shuffle or buffer_size is None:
        return dataset_.batch(step_size, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).batch(step_size, 
                                                                                          num_parallel_calls=tf.data.AUTOTUNE)

def load_cfsw_shards(start_shard, endpoint, step_size = step_size, **kwargs):
    return load_fs_shards(start_shard, endpoint,
                          image_in = base_cfsw + 'cfsw_128/',
                          geom_in = base_cfsw + 'cfsw_geom/',
                          label_in = base_cfsw + 'cfsw_labels/',
                          step_size = step_size, image_tag = "_128", 
                          file_size=256, buffer_size = shuffle_buffer_size, **kwargs)

def load_cfswp_shards(start_shard, endpoint, step_size = step_size, **kwargs):
    return load_fs_shards(start_shard, endpoint,
                          image_in = base_cfswp + 'cfswp_128/',
                          geom_in = base_cfswp + 'cfswp_geom/',
                          label_in = base_cfswp + 'cfswp_labels/',
                          step_size = step_size, image_tag = '',
                          file_size=512, buffer_size=shuffle_buffer_size, **kwargs)

def split_cfswp_shards(new_shard_size = 256, existing_shards = cfswp_shards):
    global cfswp_shard_size, cfswp_shards, total_shards
    assert cfswp_shard_size % new_shard_size == 0
    shard_factor = cfswp_shard_size // new_shard_size
    for k in range(existing_shards):
        rgb_a, flow_a, geom_a, rgb_b, flow_b, geom_b, label_seqs, label_lengths, frame_counts = load_cfswp_shards(k,k+1,return_tf_datasets =False)
        volumes = frame_counts*128*128
        vol_offsets = np.pad(volumes.cumsum(), pad_width=(1,0)) # pad a zero at start
        geom_offsets = 21*np.pad(frame_counts.reshape(-1,shard_factor).cumsum(), pad_width=(1,0))
        for j in range(shard_factor):
            suffix = str(k + j * existing_shards) + '.npy'
            low, high = j*new_shard_size, (j+1)*new_shard_size
            np.save(base_cfswp + 'cfswp_128/train_img_a_' + suffix, rgb_a[vol_offsets[low]:vol_offsets[high]])
            np.save(base_cfswp + 'cfswp_128/train_img_b_' + suffix, rgb_b[vol_offsets[low]:vol_offsets[high]])
            np.save(base_cfswp + 'cfswp_128/train_flow_a_' + suffix, flow_a[vol_offsets[low]:vol_offsets[high]])
            np.save(base_cfswp + 'cfswp_128/train_flow_b_' + suffix, flow_b[vol_offsets[low]:vol_offsets[high]])
            np.save(base_cfswp + 'cfswp_geom/train_geom_img_a_' + suffix, geom_a[geom_offsets[low]:geom_offsets[high],0:21])
            np.save(base_cfswp + 'cfswp_geom/train_geom_img_b_' + suffix, geom_b[geom_offsets[low]:geom_offsets[high],0:21])
            np.save(base_cfswp + 'cfswp_geom/train_geom_wrl_a_' + suffix, geom_a[geom_offsets[low]:geom_offsets[high],21:42])
            np.save(base_cfswp + 'cfswp_geom/train_geom_wrl_b_' + suffix, geom_b[geom_offsets[low]:geom_offsets[high],21:42])
            np.save(base_cfswp + 'cfswp_labels/train_label_seqs_' + suffix, label_seqs[low:high])
            np.save(base_cfswp + 'cfswp_labels/train_seq_lengths_' + suffix, label_lengths[low:high])
            np.save(base_cfswp + 'cfswp_labels/train_frame_counts_' + suffix, frame_counts[low:high])
    cfswp_shard_size = new_shard_size
    total_shards += (shard_factor-1)*cfswp_shards
    cfswp_shards = existing_shards * shard_factor
    return rgb_a, flow_a, geom_a, rgb_b, flow_b, geom_b, label_seqs, label_lengths, frame_counts

def process_fs_step(input_step, label_step, out_depths, return_lists=False, step_size=step_size, is_training=True):
    # input_step has frame sequences [step_size, frame_count, height, width, channels]
    #   organised as ((rgb_a, flow_a), (rgb_b, flow_b))
    # label_step has (seq labels as SparseTensors, label_lengths)
    
    encoder_results_a = testSpell3dEncode(input_step[0], is_training=False, return_dict=True)
    encoder_results_b = testSpell3dEncode(input_step[1], is_training=False, return_dict=True)
    # index 0 is rgb output, index 1 is flow output
    encoder_results_a[0].update({key:tf.stop_gradient(encoder_results_a[0][key]) for key in i3d_endpoints})
    encoder_results_a[1].update({key:tf.stop_gradient(encoder_results_a[1][key]) for key in i3d_endpoints})
    encoder_results_b[0].update({key:tf.stop_gradient(encoder_results_b[0][key]) for key in i3d_endpoints})
    encoder_results_b[1].update({key:tf.stop_gradient(encoder_results_b[1][key]) for key in i3d_endpoints})
    
    results_depth = tf.math.reduce_max(out_depths)
    encoder_results_a[0]['conv_2d'], encoder_results_a[1]['conv_2d'] = testSpell2dEncode(input_step[0])
    encoder_results_b[0]['conv_2d'], encoder_results_b[1]['conv_2d'] = testSpell2dEncode(input_step[1])
    pool_dict(encoder_results_a[0], encoder_endpoints, sizes_dict=target_sizes)
    pool_dict(encoder_results_a[1], encoder_endpoints, sizes_dict=target_sizes)
    pool_dict(encoder_results_b[0], encoder_endpoints, sizes_dict=target_sizes)
    pool_dict(encoder_results_b[1], encoder_endpoints, sizes_dict=target_sizes)
    
    attender_results_a = testSpellAttend(input_step[0][2], is_training=is_training)
    attender_results_b = testSpellAttend(input_step[1][2], is_training=is_training)
    
    for j in [0,1]:
        encoder_results_a[j].update({key:tf.math.multiply(encoder_results_a[j][key], 
                                                          attender_results_a[j][key])                                               
                                     for key in encoder_endpoints})
        encoder_results_b[j].update({key:tf.math.multiply(encoder_results_b[j][key], 
                                                          attender_results_b[j][key]) 
                                 for key in encoder_endpoints})
    
    encoder_projections_a = testSpellProject(encoder_results_a, results_depth)
    encoder_projections_b = testSpellProject(encoder_results_b, results_depth)
    
    cogitator_results_a = testSpellCogitate(tf.concat([encoder_projections_a[key] for key in encoder_endpoints], axis=-1))
    cogitator_results_b = testSpellCogitate(tf.concat([encoder_projections_b[key] for key in encoder_endpoints], axis=-1))
    
    pool_dict(encoder_results_a[0], encoder_endpoints, uniform_size = 1)
    pool_dict(encoder_results_a[1], encoder_endpoints, uniform_size = 1)
    pool_dict(encoder_results_b[0], encoder_endpoints, uniform_size = 1)
    pool_dict(encoder_results_b[1], encoder_endpoints, uniform_size = 1)
    
    decoder_input_a = tf.concat([cogitator_results_a, tf.concat([tf.squeeze(encoder_results_a[j][key], [-2,-3])
                                                        for key in encoder_endpoints for j in [0,1]], axis=-1)],
                                axis=-1)
    decoder_input_b = tf.concat([cogitator_results_b, tf.concat([tf.squeeze(encoder_results_b[j][key], [-2,-3])
                                                        for key in encoder_endpoints for j in [0,1]], axis=-1)],
                                axis=-1)

    decoder_results_list_a = testSpellDecode(decoder_input_a, is_training=is_training)
    decoder_results_list_b = testSpellDecode(decoder_input_b, is_training=is_training)
    
    #if return_lists:
    #    return decoder_results_list_a, decoder_results_list_b, out_depths
    loss_a = tf.nn.ctc_loss(label_step[0], # tensor of shape [step_size, max_label_seq_length] or SparseTensor
                         decoder_results_list_a, # tensor of shape [frames, step_size, num_labels] : prob. of each character at each time step 
                         #label_step[1],  # tensor of shape [step_size], or None if labels=SparseTensor
                            None,
                         out_depths,  # tensor of shape [step_size] "Length of input sequence in logits."
                         blank_index=0,
                         logits_time_major=False) # flip to swap first two axes of logits tensor
                         
    loss_b = tf.nn.ctc_loss(label_step[0],
                         decoder_results_list_b, 
                         #label_step[1],  
                            None,
                         out_depths,
                         blank_index=0,
                         logits_time_major=False)
    
    return tf.math.minimum(loss_a, loss_b)


# In[10]:


@tf.function
def inner_loop(input_step, label_step):             
    loss_ = tf.function(process_fs_step)(input_step, label_step, out_depths)
    loss_sum = tf.math.reduce_sum(loss_, keepdims=True)
    grad_decode = tf.gradients(loss_, testSpellDecode.trainable_variables)
    grad_cogitate = tf.gradients(loss_, testSpellCogitate.trainable_variables)
    grad_project = tf.gradients(loss_, testSpellProject.trainable_variables)
    grad_attend = tf.gradients(loss_, testSpellAttend.trainable_variables)
    grad_encode = tf.gradients(loss_, testSpell2dEncode.trainable_variables)
        
    return grad_decode, grad_cogitate, grad_project, grad_attend, grad_encode, loss_sum


# In[ ]:


# training loop: cycle through shards, loading and presenting to the net
epochs = 10
max_frames=115

# uncomment to take out the cfswp data
#cfswp_shards=24
#total_shards=cfsw_shards

run_id = rng.integers(0, 2147483647, dtype=np.int32)
logfile = 'file://run_'+str(run_id)+'.log'
tf.print('Saving to log '+logfile)
#stats_out = tf.summary.create_file_writer('temp3/', name=str(run_id), flush_millis = 30000)

step_timings = tf.Variable(initial_value=tf.zeros((2), dtype=tf.float64), trainable=False)
shard_timings = tf.Variable(initial_value=tf.zeros((2), dtype=tf.float64), trainable=False)
epoch_timings = tf.Variable(initial_value=tf.zeros((2), dtype=tf.float64), trainable=False)
shard_loss = tf.Variable(initial_value=tf.zeros(1), trainable=False)
epoch_loss = tf.Variable(initial_value=tf.zeros(1), trainable=False)
global_steps = tf.Variable(initial_value=tf.zeros(1, dtype=tf.int64), trainable=False)
out_depths = tf.Variable(initial_value = tf.zeros(step_size, dtype=tf.int32))
label_depths = tf.Variable(initial_value = tf.zeros(step_size, dtype=tf.int32))
train_shard = None
results_depth=0

for epoch in range(epochs):
    tf.print("\nStart of epoch " + str(epoch), output_stream=logfile)
    epoch_loss.assign(tf.zeros(1))
    epoch_timings.assign(tf.zeros((2), dtype=tf.float64))
    
    for k in rng.permutation(total_shards):
        shard_loss.assign(tf.zeros(1))
        shard_timings.assign(tf.zeros((2), dtype=tf.float64))
        del train_shard
        timings_temp = tf.timestamp()
        if k < cfsw_shards:
            train_shard = load_cfsw_shards(k, k+1, step_size = step_size)
            shard_size = cfsw_shard_size
        else:
            k = k - cfsw_shards
            train_shard = load_cfswp_shards(k, k+1, step_size=step_size)
            shard_size = cfswp_shard_size
        shard_timings[0].assign(tf.timestamp()-timings_temp)
        tf.print('Loaded k =', str(k), ', time', shard_timings[0], output_stream=logfile)
        for step, (input_step, label_step) in enumerate(train_shard):
            timings_temp = tf.timestamp()
            max_step_frames = 0
            for j in range(step_size):
                # since this length depends only on the depth of the input and the i3d net endpoint,
                # both A and B have the same output length, even if the frame sizes are different
                out_depths[j].assign(np.int32(np.ceil(input_step[0][0][j].shape[0]/2)))
                max_step_frames = max(max_step_frames, tf.math.reduce_max(input_step[0][0][j].shape[0]))
            if tf.math.reduce_min(out_depths-tf.cast(label_step[1], tf.int32)) < 0 or max_step_frames > max_frames:
                tf.print(out_depths, label_step[1], 'skipped', sep=', ', output_stream=logfile)
                continue
            tf.print(out_depths, label_step[1], sep=', ', end=' ', output_stream=logfile)
            results_depth = tf.math.reduce_max(out_depths)
            try:
                grad_decode, grad_cogitate, grad_project, grad_attend, grad_encode, loss_sum = inner_loop(input_step, label_step)  
                step_timings[1].assign(tf.timestamp()-timings_temp)
                shard_loss.assign_add(loss_sum)
                epoch_loss.assign_add(loss_sum)

                var_list_decay(testSpellDecode.trainable_variables, l2_coef)
                var_list_decay(testSpellCogitate.trainable_variables, l2_coef)
                var_list_decay(testSpellProject.trainable_variables, l2_coef)
                var_list_decay(testSpellAttend.trainable_variables, l2_coef)
                var_list_decay(testSpell2dEncode.trainable_variables, l2_coef)
                opt.apply_gradients(zip(grad_decode, testSpellDecode.trainable_variables))
                opt.apply_gradients(zip(grad_cogitate, testSpellCogitate.trainable_variables))
                opt.apply_gradients(zip(grad_project, testSpellProject.trainable_variables))
                opt.apply_gradients(zip(grad_attend, testSpellAttend.trainable_variables))
                opt.apply_gradients(zip(grad_encode, testSpell2dEncode.trainable_variables))                
            
                tf.print(step_timings[1], output_stream=logfile)
                #write_var_list_stats(grad_decode, 'grad_decode', stats_out, global_steps[0])
                if step % 16 == 0:
                    tf.print('\nStep ' + str(epoch) + '/' + str(k) + '/' + str(step) + ': ', output_stream=logfile)
                    tf.print(get_var_list_stats(grad_decode), output_stream=logfile)
                    tf.print(get_var_list_stats(grad_cogitate), output_stream=logfile)
                    tf.print(get_var_list_stats(grad_project), output_stream=logfile)            
                    tf.print(get_var_list_stats(grad_attend), output_stream=logfile)
                    tf.print(get_var_list_stats(grad_encode), output_stream=logfile)
                    tf.print('\n', output_stream=logfile)

                shard_timings.assign_add(step_timings)
                global_steps.assign_add(tf.ones((1), dtype=tf.int64))
            
            except tf.errors.ResourceExhaustedError as e:
                tf.print("ResourceExhaustedError raised: ", e.error_code, e.message, e.experimental_payloads,
                        output_stream=logfile)
                #max_frames -= max_frames // 20
                max_frames = min(max_step_frames-1, max_frames)
                tf.print('Set max_frames = ', max_frames, output_stream=logfile)
                if max_frames < 70:
                    raise e
                tf.print("ResourceExhaustedError raised. Attempting to continue with frame cap", max_frames)
                reboot_network(save_path)
                continue
                 
        save_network(save_path)
        tf.print('\nShard ' + str(epoch)+"/"+str(k) +': loss ', shard_loss/shard_size, 'times: ', shard_timings/shard_size,
                 summarize = -1, output_stream=logfile)
        
        epoch_timings.assign_add(shard_timings)
    tf.print('\nEpoch ' + str(epoch) + ": loss ", epoch_loss/total_examples, 'times: ', epoch_timings/total_examples,
            summarize = -1, output_stream=logfile)
