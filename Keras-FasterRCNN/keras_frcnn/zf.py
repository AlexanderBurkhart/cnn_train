# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense, Input, Conv2D, Convolution2D, MaxPooling2D, Activation, MaxPool2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length/32 #figure out why 32 and img_input cant be odd cuz cant devide whole by 2

    return get_output_length(width), get_output_length(height)
    #return get_output_length(width), get_output_length(height)    


def nn_base(input_tensor=None, trainable=False):
    
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # First convolutional Layer (96x7x7)
    z = Conv2D(96, (7,7), strides=(2,2))(img_input)
    z = ZeroPadding2D(padding = (1,1))(z)
    z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
    z = BatchNormalization(axis=3)(z)
    
    # Second convolutional Layer (256x5x5)
    z = Convolution2D(256, (5,5), strides=(4,4), activation="relu")(z)
    z = ZeroPadding2D(padding = (1,1))(z)
    z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
    z = BatchNormalization(axis=3)(z)
    
    # Rest 3 convolutional layers
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(512, (3,3), strides=(1,1), activation="relu")(z)
    
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(1024, (3,3), strides=(1,1), activation="relu")(z)
    
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(512, (3,3), strides=(1,1), activation="relu")(z)
    
    #z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)

    return z

# # Block 1
   # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
   # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

   # # Block 2
   # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
   # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

   # # Block 3
   # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
   # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
   # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

   # # Block 4
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

   # # Block 5
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
   # # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

   # return x


def rpn(base_layers, num_anchors):
    z = ZeroPadding2D(padding = (1,1))(base_layers)
    z = Convolution2D(1024, (3,3), strides=(1,1), activation="relu", kernel_initializer='normal', name='rpn_conv1')(z)

    z_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(z)
    z_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(z)

    return [z_class, z_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois, 7, 7, 512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 512, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


