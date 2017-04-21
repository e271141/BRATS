# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:26:02 2017

@author: DELL
"""

# -*- coding:utf -*-
from __future__ import division
import numpy as np
import random
import json
import h5py
from patch_library import PatchLibrary, buffered_gen_mp
from glob import glob
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from skimage.segmentation import mark_boundaries
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, Reshape, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import *
import os
import keras.utils.visualize_util as vis_util
import networks
from keras.optimizers import Adam, SGD, RMSprop


ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/zang/'
MODEL_PATH = ORI_PATH + 'models/'
LABEL_PATH = ORI_PATH + 'labels/'

class SegmentationModel(object):
    def __init__(self, n_epoch=10, n_chan=4, batch_size=100, loaded_model=False, 
        architecture=None, w_reg=0.01, n_filters=[64,128,128,128], k_dims = [7,5,5,3], activation = 'relu'):
        '''
        A class for compiling/loading, fitting and saving various models, viewing segmented images and analyzing results
        INPUT   (1) int 'n_epoch': number of eopchs to train on. defaults to 10
                (2) int 'n_chan': number of channels being assessed. defaults to 4
                (3) int 'batch_size': number of images to train on for each batch. defaults to 128
                (4) bool 'loaded_model': True if loading a pre-existing model. defaults to False
                (5) str 'architecture': type of model to use, options = single, dual, or two_path. defaults to single (only currently optimized version)
                (6) float 'w_reg': value for l1 and l2 regularization. defaults to 0.01
                (7) list 'n_filters': number of filters for each convolutional layer (4 total)
                (8) list 'k_dims': dimension of kernel at each layer (will be a k_dim[n] x k_dim[n] square). Four total.
                (9) string 'activation': activation to use at each convolutional layer. defaults to relu.
        '''
        self.n_epoch = n_epoch
        self.n_chan = n_chan
        self.batch_size = batch_size
        self.architecture = architecture
        self.loaded_model = loaded_model
        self.w_reg = w_reg
        self.n_filters = n_filters
        self.k_dims = k_dims
        #self.activation = activation
        #activation=str(raw_input('Which model architecture(single、two_path、dual) are you willing to use? '))
        self.activation =activation
        if not self.loaded_model:
            if self.architecture == 'two_path':
                self.model_comp = networks.comp_two_path()
            elif self.architecture == 'dual':
                self.model_comp = networks.comp_double()
            elif self.architecture == 'fcn':
                self.model_comp = networks.comp_fcn_model()
        else:
            model = self.architecture
            self.model_comp = self.load_model_weights(model)

    def load_model_weights(self, model_name):
        '''
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print 'Loading model {}'.format(model_name)
        model = MODEL_PATH + '{}.json'.format(model_name)
        weights = MODEL_PATH + '{}.hdf5'.format(model_name)
        with open(model) as f:
            m = f.next()
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)

        print 'Done.'
        return model_comp

    def save_model(self, model_name):
        '''
        INPUT string 'model_name': name to save model and weigths under, including filepath but not extension
        Saves current model as json and weigts as h5df file
        '''
        model = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        #json_string = self.model_comp.to_json()
        model_json = self.model_comp.to_json()

        #======================== current_dir ============================#
        current_dir = os.path.dirname(os.path.realpath(__file__))  #current_dir表示当前路径

        #======================  save model structure  ============================#
        img_path = os.path.join(current_dir, "{}.png".format(model_name))
        #print img_path
        vis_util.plot(self.model_comp, to_file=img_path, show_shapes=True)
        self.model_comp.summary()#打印出模型概况

        self.model_comp.save_weights(weights)  #保存网络模型的权重

        with open(model, 'w') as f:          
            json.dump(model_json, f)   #将网络模型编码到 .json文件中

    def class_report(self, X_test, y_test):
        '''
        returns skilearns test report (precision, recall, f1-score)
        INPUT   (1) list 'X_test': test data of 4x33x33 patches
                (2) list 'y_test': labels for X_test
        OUTPUT  (1) confusion matrix of precision, recall and f1 score
        '''
        y_pred = self.model_load.predict_class(X_test)
        print classification_report(y_pred, y_test)

    def predict_image(self, test_img, show=False):
        '''
        predicts classes of input image
        INPUT   (1) str 'test_image': filepath to image to predict on
                (2) bool 'show': True to show the results of prediction, False to return prediction
        OUTPUT  (1) if show == False: array of predicted pixel classes for the center 208 x 208 pixels
                (2) if show == True: displays segmentation results
        '''
        imgs = io.imread(test_img).astype('float').reshape(4,240,240)
        plist = [];

        # create patches from an entire slice
        for img in imgs:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (33,33)) #Reshape a 2D image into a collection of patches
            plist.append(p)
           
        patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

        # predict classes of each pixel based on model
        # Generate class predictions for the input samples batch by batch, full_pred.shape=(43264,)
        full_pred = self.model_comp.predict(patches, batch_size=200, verbose=0)
        
        # Record the category of the highest possiblility as feature
        # Create full slice feature map
        feature_map = np.zeros(full_pred.shape[0])
        for i in xrange(full_pred.shape[0]):
            feature_map[i] = np.argmax(full_pred[i])
        fp1 = feature_map.reshape(208,208)

        label0 = len(np.argwhere(feature_map==0))
        label1 = len(np.argwhere(feature_map==1))
        label2 = len(np.argwhere(feature_map==2))
        label3 = len(np.argwhere(feature_map==3))
        label4 = len(np.argwhere(feature_map==4))
        print label0, label1, label2, label3, label4

        return fp1

    def show_segmented_image(self, test_img, modality='t1c', show = True):
        '''
        Creates an image of original brain with segmentation overlay
        INPUT   (1) str 'test_img': filepath to test image for segmentation, including file extension
                (2) str 'modality': imaging modelity to use as background. defaults to t1c. options: (flair, t1, t1c, t2)
                (3) bool 'show': If true, shows output image. defaults to False.
        OUTPUT  (1) if show is True, shows image of segmentation results
                (2) if show is false, returns segmented image.
        '''
        modes = {'flair':0, 't1':1, 't1c':2, 't2':3}

        segmentation = self.predict_image(test_img, show=False)
        img_mask = np.pad(segmentation, (16,16), mode='edge') #np.pad边界扩展，即加行加列
        zeros = np.argwhere(img_mask ==0)
        ones = np.argwhere(img_mask == 1)
        twos = np.argwhere(img_mask == 2)
        threes = np.argwhere(img_mask == 3)
        fours = np.argwhere(img_mask == 4)

        test_im = io.imread(test_img)
        # overlay = mark_boundaries(test_back, img_mask)

        #img_as_float : Convert an image to floating point format, with values in [0, 1]
        gray_img = img_as_float(test_im)

        # adjust gamma of image
        #adjust_gamma : 对原图像做幂运算，g>1,新图像比原图像暗；g<1,新图像比原图像亮，这里g==0.65
        image = adjust_gamma(color.gray2rgb(gray_img), 0.65)
        sliced_image = image.copy()
        red_multiplier = [1, 0.2, 0.2]
        yellow_multiplier = [1,1,0.25]
        green_multiplier = [0.35,0.75,0.25]
        blue_multiplier = [0,0.25,0.9]
        print 'lenght of zeros:{}'.format(len(zeros))
        print 'length of ones:{}'.format(len(ones))
        print 'length of twos:{}'.format(len(twos))
        print 'length of threes:{}'.format(len(threes))
        print 'length of fours:{}'.format(len(fours))
        print 'img_mask.shape:',img_mask.shape

        # change colors of segmented classes
        # for i in xrange(len(ones)):
        #     sliced_image[ones[i][0]][ones[i][1]] = red_multiplier
        # for i in xrange(len(twos)):
        #     sliced_image[twos[i][0]][twos[i][1]] = green_multiplier
        # for i in xrange(len(threes)):
        #     sliced_image[threes[i][0]][threes[i][1]] = blue_multiplier
        # for i in xrange(len(fours)):
        #     sliced_image[fours[i][0]][fours[i][1]] = yellow_multiplier
        
        name = test_img.split('/')[8]
        labels = io.imread(LABEL_PATH+name).astype(int)
        label_zeros = np.argwhere(labels == 0)
        label_ones = np.argwhere(labels == 1)
        label_twos = np.argwhere(labels == 2)
        label_threes = np.argwhere(labels == 3)
        label_fours = np.argwhere(labels == 4)
        print 'length of label_zeros:{}'.format(len(label_zeros))
        print 'length of label_ones:{}'.format(len(label_ones))
        print 'length of label_twos:{}'.format(len(label_twos))
        print 'length of label_threes:{}'.format(len(label_threes))
        print 'length of label_fours:{}'.format(len(label_fours))
        print 'labels.shape:',labels.shape

        # for i in xrange(len(label_ones)):
        #     sliced_image[label_ones[i][0]][label_ones[i][1]] = red_multiplier
        # for i in xrange(len(label_twos)):
        #     sliced_image[label_twos[i][0]][label_twos[i][1]] = green_multiplier
        # for i in xrange(len(label_threes)):
        #     sliced_image[label_threes[i][0]][label_threes[i][1]] = blue_multiplier
        # for i in xrange(len(label_fours)):
        #     sliced_image[label_fours[i][0]][label_fours[i][1]] = yellow_multiplier

        return img_mask, labels

    def intersection(self, predict, label):
        A = np.array(predict)
        B = np.array(label)
        aset = set([tuple(x) for x in A])
        bset = set([tuple(x) for x in B])
        return np.array([x for x in aset & bset])

    def union(self, predict, label):
        A = np.array(predict)
        B = np.array(label)
        aset = set([tuple(x) for x in A])
        bset = set([tuple(x) for x in B])
        return np.array([x for x in aset | bset])

    def get_dice_coef(self, test_img):
        '''
        1: Non-brain, non-tumor, necrosis, cyst, hemorrhage
        2: Surrounding edema
        3: Non-enhancing tumor
        4: enhancing tumor core
        0: everything else

        The complete tumor region (including all four tumor structures). 
        The core tumor region (including all tumor structures exept “edema”). 
        The enhancing tumor region (including the “enhanced tumor”structure).
        '''
        
        segmentation = self.predict_image(test_img, show=True)
        seg_full = np.pad(segmentation, (16,16), mode='edge')  #the model's prediction on the test set
        
        #gt = io.imread(label).astype(int)  # gt----标签图像
        label = io.imread(test_img).astype(int).reshape(4,240,240)
        name = test_img.split('/')[8]
        gt = io.imread(LABEL_PATH+name).astype(int)

        # complete tumor
        tumor_gt = np.argwhere(gt != 0)
        tumor_seg = np.argwhere(seg_full != 0)
        n_tumor_gt = np.argwhere(gt==0)
        n_tumor_seg = np.argwhere(seg_full == 0)

        #edema 
        edema_gt = np.argwhere(gt == 2)
        edema_seg = np.argwhere(seg_full == 2)
            
        # enhancing tumor
        adv_gt = np.argwhere(gt == 4)
        adv_seg = np.argwhere(seg_full == 4)
        n_adv_gt = np.argwhere(gt != 4)
        n_adv_seg = np.argwhere(seg_full != 4)

        # core tumor 1,3,4
        noadv_gt = np.argwhere(gt == 3)
        necrosis_gt = np.argwhere(gt == 1)
        core_gt = np.append(adv_gt, noadv_gt, axis = 0)
        core_gt = np.append(core_gt, necrosis_gt, axis = 0)
        n_core_gt = np.append(n_tumor_gt, edema_gt, axis = 0)

        noadv_seg = np.argwhere(seg_full == 3)
        necrosis_seg = np.argwhere(seg_full == 1)
        core_seg = np.append(noadv_seg, adv_seg, axis = 0)
        core_seg = np.append(core_seg, necrosis_seg, axis = 0)
        n_core_seg = np.append(n_tumor_seg, edema_seg, axis = 0)

        tumor = self.intersection(tumor_gt, tumor_seg)
        adv = self.intersection(adv_gt, adv_seg)
        core = self.intersection(core_gt, core_seg)

        n_tumor = self.intersection(n_tumor_gt, n_tumor_seg)
        n_adv = self.intersection(n_adv_gt, n_adv_seg)
        n_core = self.intersection(n_core_gt, n_core_seg)

        return [len(tumor), len(tumor_gt), len(tumor_seg), len(n_tumor),len(n_tumor_gt), len(n_tumor_seg)], \
            [len(adv), len(adv_gt), len(adv_seg), len(n_adv), len(n_adv_gt), len(n_adv_seg)], \
            [len(core), len(core_gt), len(core_seg), len(n_core), len(n_core_gt), len(n_core_seg)]

if __name__ == '__main__':
    nb_epoch=10
    batch_size=100
    # train_data = glob('/mnt/filsystem1/code/dsb2017/code/zang/train_set/**')
    # val_data = glob('/mnt/filsystem1/code/dsb2017/code/zang/val_set/**')
    # patches = PatchLibrary((33,33), train_data, val_data, batch_size)
    # train_gen = buffered_gen_mp(patches.make_patches(data_set='train'), buffer_size=1)
    # val_gen = buffered_gen_mp(patches.make_patches(data_set='valedation'), buffer_size=1)
    '''
    X1 = np.load('/mnt/filsystem1/code/dsb2017/code/zang/gen_train.npy')
    Y = np.load('/mnt/filsystem1/code/dsb2017/code/zang/gen_labels.npy')

    model = SegmentationModel(n_epoch=10, batch_size=100, architecture='two_path')
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    model.model_comp.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="./models/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,period=10)
    # for i in range(10):
    #     model.model_comp.fit_generator(train_gen, samples_per_epoch = batch_size*700, 
    #         nb_epoch = nb_epoch, verbose = 1, validation_data = val_gen, nb_val_samples = batch_size*150)

    model.model_comp.fit(X1, Y, batch_size=200, nb_epoch=10, validation_split=0.2, callbacks=[checkpointer])
    model.save_model('models/two_path')
    '''

    tests = glob('/mnt/filsystem1/code/dsb2017/code/zang/val_set/**')

    model =SegmentationModel(loaded_model=True,architecture='two_path')

    '''
    for i in xrange(50,100):
        ori = io.imread(tests[i])
        img = ori.reshape(4,240,240)
        predict, label= model.show_segmented_image(tests[i])
        print tests[i]
        plt.figure(13)
        plt.subplot(131)
        plt.imshow(predict)
        plt.subplot(132)
        plt.imshow(label)
        plt.subplot(133)
        plt.imshow(img[1], cmap='gray')
        plt.show()
    '''
    
    n_full=np.zeros(6); n_core=np.zeros(6); n_advancing=np.zeros(6)
    for i in xrange(len(tests)):
        full, core, advancing = model.get_dice_coef(tests[i])
        n_full += full
        n_core += core
        n_advancing += advancing
    print 'full_tumor:{}\n core_tumor:{}\n advancing_tumor:{}\n'.format(n_full, n_core, n_advancing)
    

    d_full= 2*float(n_full[0]) / float(n_full[1]+n_full[2])
    d_core= 2*float(n_core[0]) / float(n_core[1]+n_core[2])
    d_advancing= 2*float(n_advancing[0]) / float(n_advancing[1]+n_advancing[2])

    sen_full = float(n_full[0]) / float(n_full[1])
    sen_core = float(n_core[0]) / float(n_core[1])
    sen_advancing = float(n_advancing[0]) / float(n_advancing[1])

    spec_full = float(n_full[3]) / float(n_full[4])
    spec_core = float(n_core[3]) / float(n_core[4])
    spec_advancing = float(n_advancing[3]) / float(n_advancing[4])

    print ' '
    print 'Region_________________| Dice Coefficient'
    print 'Complete Tumor_________| {0:.2f}'.format(d_full)
    print 'Core Tumor_____________| {0:.2f}'.format(d_core)
    print 'Enhancing Tumor________| {0:.2f}'.format(d_advancing)

    print 'Region_________________| Sensitivity'
    print 'Complete Tumor_________| {0:.2f}'.format(sen_full)
    print 'Core Tumor_____________| {0:.2f}'.format(sen_core)
    print 'Enhancing Tumor________| {0:.2f}'.format(sen_advancing)

    print 'Region_________________| Specificity'
    print 'Complete Tumor_________| {0:.4f}'.format(spec_full)
    print 'Core Tumor_____________| {0:.4f}'.format(spec_core)
    print 'Enhancing Tumor________| {0:.4f}'.format(spec_advancing)
