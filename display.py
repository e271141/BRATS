from __future__ import division
import os
import numpy as np
from skimage import io
import nipy as ni
import matplotlib.pyplot as plt
from skimage import io

PATH = '/mnt/filsystem1/code/dsb2017/code/zang/brats2013/'

# label = ni.load_image(PATH + 'brats_2013_pat0001_1_OT.nii.gz')
# flair = ni.load_image(PATH + 'brats_2013_pat0001_1_Flair_sus_nml.nii.gz')
# T1 = ni.load_image(PATH + 'brats_2013_pat0001_1_T1_sus_nml.nii.gz')
# T1c = ni.load_image(PATH + 'brats_2013_pat0001_1_T1c_sus_nml.nii.gz')
# T2 = ni.load_image(PATH + 'brats_2013_pat0001_1_T2_sus_nml.nii.gz')

label = io.imread('/mnt/filsystem1/code/dsb2017/code/zang/labels/pat0006_108.png')
compare = io.imread('/mnt/filsystem1/code/dsb2017/code/zang/val_set/pat0006_108.png').astype('float').reshape(4,240,240)
sample = compare[2]
sample_set = np.argwhere(compare[0] != 0)
print sample_set.shape

plt.subplot(121)
plt.imshow(compare[2], cmap='gray')
plt.subplot(122)
plt.imshow(label, cmap='gray')
plt.show()

labels = np.load('/mnt/filsystem1/code/dsb2017/code/zang/labels.npy')
class1 = np.argwhere((labels == 1))
label0 = len(np.argwhere(class1[:,1]==0))
label1 = len(np.argwhere(class1[:,1]==1))
label2 = len(np.argwhere(class1[:,1]==2))
label3 = len(np.argwhere(class1[:,1]==3))
label4 = len(np.argwhere(class1[:,1]==4))
print labels.shape
print label0, label1, label2, label3, label4
print labels[0]

# image = np.load('/mnt/filsystem1/code/dsb2017/code/zang/gen_train.npy')
# print image.shape

'''
plt.figure(23)

plt.subplot(231)
plt.imshow(label, cmap='gray')

plt.subplot(232)
plt.imshow(temp[1], cmap='gray')

plt.subplot(233)
plt.imshow(temp[2], cmap='gray')

plt.subplot(234)
plt.imshow(temp[3], cmap='gray')

plt.subplot(235)
plt.imshow(temp[0], cmap='gray')

# plt.subplot(236)
# plt.imshow(compare, cmap='gray')

plt.show()
'''