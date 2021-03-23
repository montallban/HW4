
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential
import random
import re

import png
import sklearn.metrics
import os
import fnmatch
import re
import numpy as np
from core50 import *

# class CNN_Classifier():
#     def __init__(self, self.image_size, self.nchannels,
#                 self.conv_layers=conv_layers,
#                 self.conv_size=args.conv_size,
#                 self.filters=args.conv_nfilters,
#                 self.dense_layers=dense_layers,
#                 hidden=args.hidden,
#                 self.p_dropout=args.dropout,
#                 self.lambda_l2=args.L2_regularizer,
#                 self.lrate=args.lrate, n_classes=3):

# def create_cnn_classifier_network(self.image_size, self.nchannels,
#                                         conv_layers=self.conv_layers,
#                                         conv_size=self.conv_size,
#                                         filters=self.conv_nfilters,
#                                         dense_layers=self.dense_layers,
#                                         hidden=self.hidden,
#                                         p_dropout=self.dropout,
#                                         lambda_l2=self.L2_regularizer,
#                                         lrate=self.lrate, self.n_classes=3):

def create_cnn_classifier_network(image_size, nchannels,
                                        conv_size,
                                        conv_nfilter,
                                        filters,
                                        dense_layers,
                                        pool,
                                        hidden,
                                        p_dropout,
                                        lambda_l2,
                                        lrate, n_classes=3):

    model = Sequential();
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))

# number of weights is kernel_size = 25+1*10 (5 * (5+1) * 10) + 500 ?
    for idx, filters in enumerate(conv_nfilters):
        kernel_size = conv_size[idx]
        name = "C" + str(idx)
        model.add(Convolution2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='valid',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name=name,
                            activation='elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))

        model.add(MaxPooling2D(pool_size=(pool[idx],pool[idx]), strides=(2,2))) # reduce dimensionality at a minimum amount
    
    model.add(Flatten())
    
    for idx, units in enumerate(dense_layers):
        name = "D" + str(idx)
        model.add(Dense(units=units,
                    activation='elu',
                    use_bias=True,
                    kernel_initializer='truncated_normal',
                    bias_initializer='zeros',
                    name=name,
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    
    model.add(Dropout(p_dropout))
    
    model.add(Dense(units=n_classes,
                activation='softmax',
                use_bias=True,
                kernel_initializer='truncated_normal',
                bias_initializer='zeros',
                name='D2',
                kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    opt=tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    # there is conv3D for volumetric data
    print(model.summary())
    
    return model
