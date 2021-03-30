
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
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



def create_cnn_classifier_network(image_size, nchannels,
                                        conv_size,
                                        conv_nfilters,
                                        conv_layers,
                                        filters,
                                        dense_layers,
                                        pool,
                                        hidden,
                                        p_dropout,
                                        lambda_l2,
                                        lrate, n_classes=3,
                                        type='basic'):

    model = Sequential();
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))

# number of weights is kernel_size = 25+1*10 (5 * (5+1) * 10) + 500 ?

    for idx, layer in enumerate(conv_layers):
        print(layer)
        filters = layer['filters']
        kernel_size = layer['kernel_size'] 
        pool_size = layer['pool_size']
        strides = layer['strides']
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
    
    for idx, layer in enumerate(dense_layers):
        print(layer)
        name = "H" + str(idx)
        units = layer['units']
        model.add(Dense(units=units,
                    activation='elu',
                    use_bias=True,
                    kernel_initializer='truncated_normal',
                    bias_initializer='zeros',
                    name=name,
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
        model.add(Dropout(p_dropout))
    
    
    model.add(Dropout(p_dropout))
    
    model.add(Dense(units=n_classes,
                activation='softmax',
                use_bias=True,
                kernel_initializer='truncated_normal',
                bias_initializer='zeros',
                name='output',
                kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    opt=tf.keras.optimizers.Adam(lr=lrate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
    # there is conv3D for volumetric data
    print(model.summary())
    
    return model


def create_deep_cnn_classifier_network(image_size, nchannels,
                                        conv_size,
                                        conv_nfilters,
                                        conv_layers,
                                        filters,
                                        dense_layers,
                                        pool,
                                        hidden,
                                        p_dropout,
                                        lambda_l2,
                                        lrate, n_classes=3,
                                        type='basic'):

    '''
    In this deep network, we add multiple convolution layers,
    whereas, in the shallow network we add single conv layers 
    before we pool, here we use 2
    '''

    model = Sequential();
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))

# number of weights is kernel_size = 25+1*10 (5 * (5+1) * 10) + 500 ?

    i = 0
    for idx, layer in enumerate(conv_layers):
        filters = layer['filters']
        kernel_size = layer['kernel_size'] 
        pool_size = layer['pool_size']
        strides = layer['strides']
        kernel_size = conv_size[idx]
        name = "C" + str(i)
        model.add(Convolution2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='valid',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name=name,
                            activation='elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
        
        i = i + 1
        name = "C" + str(i)
        
        # Next, we add an identical conv layer 
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
        i = i + 1

    model.add(Flatten())
    
    for idx, layer in enumerate(dense_layers):
        print(layer)
        name = "H" + str(idx)
        units = layer['units']
        model.add(Dense(units=units,
                    activation='elu',
                    use_bias=True,
                    kernel_initializer='truncated_normal',
                    bias_initializer='zeros',
                    name=name,
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
        model.add(Dropout(p_dropout))
    
    
    model.add(Dropout(p_dropout))
    
    model.add(Dense(units=n_classes,
                activation='softmax',
                use_bias=True,
                kernel_initializer='truncated_normal',
                bias_initializer='zeros',
                name='output',
                kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    opt=tf.keras.optimizers.Adam(lr=lrate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
    # there is conv3D for volumetric data
    print(model.summary())
    
    return model

def create_inception_network(image_size, n_channels,
                              lambda_regularization, activation='elu'):


    input_tensor = Input(shape=(image_size[0], image_size[1], n_channels),name="input")

    i1_tensor = inception_module(input_tensor, (10, (10,10), (10,10), 10), activation,
                                                    lambda_regularization, name="i1")

    i2_tensor = inception_module(i1_tensor, (10, (10,10), (10,10), 10), activation,
                                                    lambda_regularization, name="i2")
                                                
    flatten_tensor = Flatten()(i2_tensor)
                                 
    dense1_tensor = Dense(units=100, activation=activation, name = "D1") (flatten_tensor)
    dense2_tensor = Dense(units=20, activation=activation, name = "D2") (dense1_tensor)
    output_tensor = Dense(units=3, activation='sigmoid', name = "output") (dense2_tensor)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    #loss_fn = keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss='categorical_crossentropy', optimizer = opt,
                    metrics=['accuracy'])

    print(model.summary())
    return model


def inception_module(input_tensor, nfilters, activation, lambda_regularization, name):

    convA_tensor = Convolution2D(filters=nfilters[0],
                                  kernel_size=(1,1),
                                  strides=(2,2),    # reduces dimensionality by 2
                                  padding='same',
                                  name = 'convA'+name) (input_tensor)
                                  
    convB0_tensor = Convolution2D(filters=nfilters[1][0],
                                    kernel_size=(1,1),
                                    strides=(1,1),
                                    padding='same',
                                    name = 'convB0'+name) (input_tensor)
                                
    convB1_tensor = Convolution2D(filters=nfilters[1][1],
                                    kernel_size=(3,3),
                                    strides=(2,2),  # reduces dimensionality by 2
                                    padding='same',
                                    name = 'convB1'+name) (convB0_tensor)

    convC0_tensor = Convolution2D(filters=nfilters[2][0],
                                    kernel_size=(1,1),
                                    strides=(1,1),
                                    padding='same',
                                    name = 'convC0'+name) (input_tensor)

    convC1_tensor = Convolution2D(filters=nfilters[2][1],
                                    kernel_size=(5,5),
                                    strides=(2,2),  # reduces dimensionality by 2
                                    padding='same',
                                    name = 'convC1'+name) (convC0_tensor)


    max_tensor = MaxPooling2D(pool_size=(3,3),
                              strides=(1,1),
                              name='MAX_'+name,
                              padding='same')(input_tensor)

    convD1_tensor = Convolution2D(filters=nfilters[3],
                                    kernel_size=(1,1),
                                    strides=(2,2),  # reduces dimensionality by 2
                                    padding='same',
                                    name = 'convD0'+name) (max_tensor)

    output_tensor = concatenate([convA_tensor, convB1_tensor, convC1_tensor, convD1_tensor])
        
    return output_tensor

def create_multiple():

    input_tensor1 = Input(shape=(image_size[0], image_size[1], n_channels),
                            name="input1")

    input_tensor2 = Input(shape=(image_size[0], image_size[1], n_channels),
                            name="input2")        

    model = Model(ipnuts=[input_tensor1, input_tensor2],
                                        outputs=output_tensor)            