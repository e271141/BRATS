import numpy as np
import random
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import MaxoutDense
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils.visualize_util import plot
from keras.optimizers import Adam, SGD, RMSprop
import os


def comp_two_path():
    '''
    compiles two-path model, takes in a 4x33x33 patch and assesses global and local paths, then merges the results.
    '''
    print 'Compiling two-path model...'
    input_image = Input(shape=(4,33,33))
    conv1_local = Convolution2D(64, 7,7, border_mode='valid', activation='relu')(input_image)
    pool1_local = MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid')(conv1_local)
    
    conv2_local = Convolution2D(64, 3,3, border_mode='valid', activation='relu')(pool1_local)
    pool2_local = MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid')(conv2_local)

    conv1_global = Convolution2D(160, 13,13, border_mode='valid', activation='relu')(input_image)

    combine = merge([pool2_local, conv1_global], mode='concat', concat_axis=1)
    conv1_combine = Convolution2D(5, 21,21, border_mode='valid', activation='relu')(combine)
    output = Flatten()(conv1_combine)
    output = Activation('softmax')(output)

    model = Model(input_image, [output])
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print 'Done.'
    return model

def comp_local_path():
    '''
    compiles local-path model, takes in a 4x33x33 patch and assesses global and local paths, then merges the results.
    '''
    print 'Compiling local-path model...'
    input_image = Input(shape=(4,33,33))
    conv1_local = Convolution2D(64, 7,7, border_mode='valid', activation='relu')(input_image)
    pool1_local = MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid')(conv1_local)
    
    conv2_local = Convolution2D(64, 3,3, border_mode='valid', activation='relu')(pool1_local)
    pool2_local = MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid')(conv2_local)

    conv1_combine = Convolution2D(5, 21,21, border_mode='valid', activation='relu')(pool2_local)
    output = Flatten()(conv1_combine)
    output = Activation('softmax')(output)

    model = Model(input_image, [output])
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print 'Done.'
    return model


def comp_global_path():
    '''
    compiles global-path model, takes in a 4x33x33 patch and assesses global and local paths, then merges the results.
    '''
    print 'Compiling global-path model...'
    input_image = Input(shape=(4,33,33)) 
    conv2_local = Convolution2D(160, 13, 13, border_mode='valid', activation='relu')(input_image)
    conv1_combine = Convolution2D(5, 21,21, border_mode='valid', activation='relu')(conv2_local)
    output = Flatten()(conv1_combine)
    output = Activation('softmax')(output)

    model = Model(input_image, [output])
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print 'Done.'
    return model


def comp_triple():
    print 'Compiling triple model...'
    input_image = Input(shape=(4,33,33))
    conv1_local = Convolution2D(64, 7,7, border_mode='valid', activation='relu')(input_image)
    pool1_local = MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid')(conv1_local)
    
    conv2_local = Convolution2D(64, 3,3, border_mode='valid', activation='relu')(pool1_local)
    pool2_local = MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid')(conv2_local)

    conv1_third = Convolution2D(64, 9, 9, border_mode='valid', activation='relu')(input_image)
    pool1_third = MaxPooling2D(pool_size=(5,5), strides=(1,1), border_mode='valid')(conv1_third)

    conv1_global = Convolution2D(160, 13,13, border_mode='valid', activation='relu')(input_image)

    combine = merge([pool2_local, pool1_third, conv1_global], mode='concat', concat_axis=1)
    conv1_combine = Convolution2D(5, 21,21, border_mode='valid', activation='relu')(combine)
    output = Flatten()(conv1_combine)
    output = Activation('softmax')(output)

    model = Model(input_image, [output])
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print 'Done.'
    return model

def comp_triple_7():
    print 'Compiling triple_7 model...'
    input_image = Input(shape=(4,33,33))
    conv1_local = Convolution2D(64, 7,7, border_mode='valid', activation='relu')(input_image)
    pool1_local = MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid')(conv1_local)
    
    conv2_local = Convolution2D(64, 3,3, border_mode='valid', activation='relu')(pool1_local)
    pool2_local = MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid')(conv2_local)

    conv1_third = Convolution2D(64, 7, 7, border_mode='valid', activation='relu')(input_image)
    pool1_third = MaxPooling2D(pool_size=(7,7), strides=(1,1), border_mode='valid')(conv1_third)

    conv1_global = Convolution2D(160, 13,13, border_mode='valid', activation='relu')(input_image)

    combine = merge([pool2_local, pool1_third, conv1_global], mode='concat', concat_axis=1)
    conv1_combine = Convolution2D(5, 21,21, border_mode='valid', activation='relu')(combine)
    output = Flatten()(conv1_combine)
    output = Activation('softmax')(output)

    model = Model(input_image, [output])
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print 'Done.'
    return model