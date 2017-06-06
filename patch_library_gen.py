# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:38:50 2017

@author: DELL
"""
import numpy as np
import random
import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
import multiprocessing as mp
from keras.utils import np_utils
import time
import Queue
np.random.seed(5)

ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/zang/'
LABEL_PATH = ORI_PATH + 'labels/'

class PatchLibrary(object):
    def __init__(self, patch_size,train_data, val_data,num_samples):
        '''
        class for creating patches and subpatches from training data to use as input for segmentation models.
        INPUT   (1) tuple 'patch_size': size (in voxels) of patches to extract. Use (33,33) for sequential model
                (2) list 'train_data': list of filepaths to all training data saved as pngs. images should have shape (5*240,240)
                (3) int 'num_samples': the number of patches to collect from training data.
        '''
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
        self.val_data = val_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches,im_path):
        '''
        Helper function for sampling slices with evenly distributed classes
        INPUT:  (1) list 'training_images': all training images to select from
                (2) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
                (3) tuple 'patch_size': dimensions of patches to be generated defaults to 65 x 65
        OUTPUT: (1) num_samples patches from class 'class_num' randomly selected.
        '''
        h,w = self.patch_size[0], self.patch_size[1]
        patches,labels = [],np.full(num_patches, class_num, 'float')

        img = io.imread(im_path).reshape(4, 240, 240).astype('float')
        img = np.pad(img, (16,16), mode='edge')
        label = img[4]
        ct = 0
        cor = np.argwhere((label == class_num))
        while ct < num_patches:

            if len(cor) > 1:
                p = random.choice(cor)
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img[:4]])
                label = label[p[0]][p[1]]
            else:
                ct += 1
                continue

            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                continue
            patches.append(patch)

            ct += 1


        return np.array(patches), label
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

    def center_n(self, n, patches):
        '''
        Takes list of patches and returns center nxn for each patch. Use as input for cascaded architectures.
        INPUT   (1) int 'n': size of center patch to take (square)
                (2) list 'patches': list of patches to take subpatch of
        OUTPUT: list of center nxn patches.
        '''
        sub_patches = []
        for mode in patches:
            subs = np.array([patch[(self.h/2) - (n/2):(self.h/2) + ((n+1)/2),(self.w/2) - (n/2):(self.w/2) + ((n+1)/2)] for patch in mode])
            sub_patches.append(subs)
        return np.array(sub_patches)
    
    def slice_to_patches(self, filename):
        '''
        Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
        INPUT: str 'filename': path to image to be converted to patches
        OUTPUT: list of patched version of input image.
        '''
        slices = io.imread(filename).astype('float').reshape(5,240,240)[:-1]
        plist=[]
        for slice in slices:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (h,w))
            plist.append(p)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def patches_by_entropy(self, num_patches):
        '''
        Finds high-entropy patches based on label, allows net to learn borders more effectively.
        INPUT: int 'num_patches': defaults to num_samples, enter in quantity it using in conjunction with randomly sampled patches.
        OUTPUT: list of patches (num_patches, 4, h, w) selected by highest entropy
        '''
        h,w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], []
        ct = 0
        while ct < num_patches:
            #im_path = random.choice(training_images)
            im_path = random.choice(self.train_data)        
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # pick again if slice is only background
            if len(np.unique(label)) == 0:
                continue

            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            l_ent = entropy(label, disk(self.h))
            top_ent = np.percentile(l_ent, 90)

            # restart if 80th entropy percentile = 0
            if top_ent == 0:
                continue

            highest = np.argwhere(l_ent >= top_ent)
            p_s = random.sample(highest, 1)
            for p in p_s:
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                # exclude any patches that are too small
                if np.shape(patch) != (4,65,65):
                    continue
                patches.append(patch)
                labels.append(label[p[0],p[1]])
            #print '**in patches_by_entropy,patches.shape:',np.array(patches).shape #(3,4,65,65)
            #print '**in patches_by_entropy,labels.shape:',np.array(labels).shape 
            ct += 1
        return np.array(patches[:num_patches]), np.array(labels[:num_patches])

    def gen_patches(self, data_set ,entropy=False, balanced_classes=True, classes=[0,1,2,3,4]):
        '''
        Creates X and y for training CNN
        INPUT   Path of all the training data set
                INPUT   Path of all the training data set.
                Choose train/valedation set folders by typing data_set=='train/valedation'
                when calling this function.
        OUTPUT  Data generator. generates 100 training samples. 
                Currently generates unfixed size samples due to various conditions.
        '''
        if data_set=='train':
            sets = self.train_data
        elif data_set=='valedation':
            sets = self.val_data
        for candidate in sets:
            name = candidate.split('/')[8]
            h,w = self.patch_size[0], self.patch_size[1]
            img = io.imread(candidate).reshape(4, 240, 240).astype('float')
            # Record the brain tissue
            brain = np.argwhere(img[0] != 0)

            groud_truth = io.imread(LABEL_PATH+name)
            img = [np.pad(i, (16,16), mode='edge') for i in img]

            patches, labels = [], []
            for i in xrange(len(classes)):
                # Because there's too much sample for category 0
                # We choose only 0.8% samples from 0
                if i==0: percent = 0.008
                elif i==2: percent = 0.16
                elif i==4: percent = 0.375
                else: percent = 0.95
                samples = np.argwhere((groud_truth == classes[i]))
                cor = random.sample(samples, int(len(samples)*percent))

                j=i
        
                for i in xrange(len(cor)):

                    p = cor[i]
                    p_ix = (p[0]-(h/2)+16, p[0]+((h+1)/2)+16, p[1]-(w/2)+16, p[1]+((w+1)/2)+16)
                    patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img[:4]])
                    label = groud_truth[p[0]][p[1]]

                    if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > 2*(h * w):
                        #print str(p) ,patch.shape, len(np.argwhere(patch == 0))
                        continue
                    else:
                        #set 0 <= pix intensity <= 1 
                        for slice in xrange(len(patch)):
                            if np.max(patch[slice]) != 0:
                                patch[slice] /= np.max(patch[slice])

                        patches.append(patch)
                        labels.append(label)
            if len(labels)==0:
                continue
        
            X1_train=np.array(patches).reshape(len(patches), 4, self.h, self.w)
            Y_train=np.array(labels).reshape(len(patches))
            
            Y_train = np_utils.to_categorical(Y_train, 5)
            shuffle = zip(X1_train, Y_train)
            np.random.shuffle(shuffle)

            X1_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
            Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])

            yield X1_train, Y_train

    def make_patches(self, data_set ,entropy=False, balanced_classes=True, classes=[0,1,2,3,4]):
        '''
        Creates X and y for training CNN
        INPUT   Path of all the training data set.
                Choose train/valedation set folders by typing data_set=='train/valedation'
                when calling this function.
        OUTPUT  All training set are saved in .npy file.
        '''
        if data_set=='train':
            sets = self.train_data
        elif data_set=='valedation':
            sets = self.val_data
        patches, labels = [], []
        sum_zeros = np.zeros(5)
        for candidate in sets:
            name = candidate.split('/')[8]
            h,w = self.patch_size[0], self.patch_size[1]
            img = io.imread(candidate).reshape(4, 240, 240).astype('float')
            # Record the brain tissue
            brain = np.argwhere(img[0] != 0)

            groud_truth = io.imread(LABEL_PATH+name)
            img = [np.pad(i, (16,16), mode='edge') for i in img]

            patches, labels = [], []
            for i in xrange(len(classes)):
                # Because there's too much sample for category 0
                # We choose only 0.8% samples from 0
                if i==0: percent = 0.008
                elif i==2: percent = 0.16
                elif i==4: percent = 0.375
                else: percent = 0.95
                samples = np.argwhere((groud_truth == classes[i]))
                cor = random.sample(samples, int(len(samples)*percent))

                j=i
        
                for i in xrange(len(cor)):

                    p = cor[i]
                    p_ix = (p[0]-(h/2)+16, p[0]+((h+1)/2)+16, p[1]-(w/2)+16, p[1]+((w+1)/2)+16)
                    patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img[:4]])
                    label = groud_truth[p[0]][p[1]]

                    if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > 2*(h * w):
                        #print str(p) ,patch.shape, len(np.argwhere(patch == 0))
                        continue
                    else:
                        sum_zeros[j] += 1
                        #set 0 <= pix intensity <= 1 
                        for slice in xrange(len(patch)):
                            if np.max(patch[slice]) != 0:
                                patch[slice] /= np.max(patch[slice])

                        patches.append(patch)
                        labels.append(label)
            if len(labels)==0:
                continue
        print sum_zeros, sum(sum_zeros)
        
        X1_train=np.array(patches).reshape(len(patches), 4, self.h, self.w)
        Y_train=np.array(labels).reshape(len(patches))
        
        Y_train = np_utils.to_categorical(Y_train, 5)
        shuffle = zip(X1_train, Y_train)
        np.random.shuffle(shuffle)

        X1_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])

        np.save('gen_train.npy', X1_train)
        np.save('gen_labels.npy', Y_train)
           

def buffered_gen_mp(source_gen, buffer_size=1, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = mp.Queue(maxsize=buffer_size)

    def _buffered_generation_process(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                # print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                print "DEBUG: OUT OF DATA, CLOSING BUFFER"
                buffer.close() # signal that we're done putting data in the buffer
                break

            buffer.put(data)
    
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
    
    while True:
        try:
            # yield buffer.get()
            # just blocking on buffer.get() here creates a problem: when get() is called and the buffer
            # is empty, this blocks. Subsequently closing the buffer does NOT stop this block.
            # so the only solution is to periodically time out and try again. That way we'll pick up
            # on the 'close' signal.
            try:
                yield buffer.get(True, timeout=sleep_time)
            except Queue.Empty:
                if not process.is_alive():
                    break # no more data is going to come. This is a workaround because the buffer.close() signal does not seem to be reliable.

                # print "DEBUG: queue is empty, waiting..."
                pass # ignore this, just try again.

        except IOError: # if the buffer has been closed, calling get() on it will raise IOError.
            # this means that we're done iterating.
            # print "DEBUG: buffer closed, stopping."
            break
        except KeyboardInterrupt:
            buffer.close()
            break
if __name__ == '__main__':

    train_data = glob('/mnt/filsystem1/code/dsb2017/code/zang/temp/**')
    val_data = glob('/mnt/filsystem1/code/dsb2017/code/zang/test_set/**')
    patches = PatchLibrary((33,33), train_data, val_data, 160)

    patches.make_patches(data_set='train')

