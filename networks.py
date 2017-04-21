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

def comp_fcn_model():
	### Jingzhi Zang: error in dimensions
    '''
    compiles standard single model with 4 convolitional/max-pooling layers.
    '''
    print 'Compiling single model...'
    single = Sequential()
    single.add(Convolution2D(self.n_filters[0], self.k_dims[0], self.k_dims[0], 
        border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg), dim_ordering='th',input_shape=(self.n_chan,33,33)))

    single.add(Activation(self.activation))
    single.add(BatchNormalization(mode=0, axis=1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Convolution2D(self.n_filters[1], self.k_dims[1], self.k_dims[1], 
        activation=self.activation, border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg),dim_ordering='th'))

    single.add(BatchNormalization(mode=0, axis=1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Convolution2D(self.n_filters[2], self.k_dims[2], self.k_dims[2], 
        activation=self.activation, border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg),dim_ordering='th'))

    single.add(BatchNormalization(mode=0, axis=1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Convolution2D(self.n_filters[3], self.k_dims[3], self.k_dims[3], 
        activation=self.activation, border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg),dim_ordering='th'))

    single.add(Dropout(0.25))
    single.add(Flatten())
    single.add(Dense(5))
    single.add(Activation('softmax'))

    sgd = SGD(lr=0.00001, decay=0.01, momentum=0.9)
    single.compile(loss='categorical_crossentropy', optimizer='sgd')
    print 'Done.'
    return single

def comp_double():
	### Jingzhi Zang: error in dimensions
    '''
    double model. Simialar to two-pathway, except takes in a 4x33x33 patch and it's center 4x5x5 patch. merges paths at flatten layer.
    '''
    print 'Compiling double model...'
    one = Sequential()
    one.add(Convolution2D(64, 11, 11, border_mode='valid', 
        W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th', input_shape=(4,48,48)))
    one.add(Activation('relu'))
    one.add(BatchNormalization(mode=0, axis=1))
    one.add(Dropout(0.5))
    
    one.add(Convolution2D(nb_filter=128, nb_row=11, nb_col=11, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    one.add(BatchNormalization(mode=0, axis=1))
    one.add(Dropout(0.5))
    
    one.add(Convolution2D(nb_filter=128, nb_row=11, nb_col=11, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    one.add(BatchNormalization(mode=0, axis=1))
    one.add(Dropout(0.5))

    one.add(Convolution2D(nb_filter=256, nb_row=11, nb_col=11, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    one.add(BatchNormalization(mode=0, axis=1))
    one.add(Dropout(0.5))

    one.add(Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    one.add(BatchNormalization(mode=0, axis=1))
    one.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),dim_ordering='th'))
    one.add(Flatten())
    
    two=Sequential()
    two.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th',input_shape=(4,28,28)))
    two.add(BatchNormalization(mode=0, axis=1))
    two.add(MaxPooling2D(pool_size=(2,2), strides=(1,1),dim_ordering='th'))
    two.add(Convolution2D(nb_filter=72, nb_row=5, nb_col=5, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    two.add(BatchNormalization(mode=0, axis=1))
    two.add(MaxPooling2D(pool_size=(2,2), strides=(1,1),dim_ordering='th'))        
    two.add(Flatten())

    # add small patch to train on
    three = Sequential()
    three.add(Convolution2D(64, 3, 3, border_mode='valid', 
        W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th', input_shape=(4,12,12)))
    three.add(Activation('relu'))
    three.add(BatchNormalization(mode=0, axis=1))
    three.add(Dropout(0.5))
    
    three.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    three.add(BatchNormalization(mode=0, axis=1))
    three.add(Dropout(0.5))
    
    three.add(Convolution2D(nb_filter=16, nb_row=3, nb_col=3, 
        activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01),dim_ordering='th'))
    three.add(BatchNormalization(mode=0, axis=1))
    three.add(Dropout(0.5))
    three.add(Flatten())

    model = Sequential()
    # merge both paths
    model.add(Merge([one, two,three], mode='concat', concat_axis=1))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    #weights_path = '/home/zhu/2017/tumor_seg/dual_n5.hdf5'
    #model.load_weights(weights_path, by_name=True)
    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    print 'Done.'
    return mode