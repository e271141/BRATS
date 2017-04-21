from __future__ import division
import os
import numpy as np
from skimage import io
import nipy as ni
import matplotlib.pyplot as plt
from skimage import io

PATH = '/mnt/filsystem1/code/dsb2017/code/zang/brats2013/'
'''
label = ni.load_image(PATH + 'brats_2013_pat0001_1_OT.nii.gz')
flair = ni.load_image(PATH + 'brats_2013_pat0001_1_Flair_sus_nml.nii.gz')
T1 = ni.load_image(PATH + 'brats_2013_pat0001_1_T1_sus_nml.nii.gz')
T1c = ni.load_image(PATH + 'brats_2013_pat0001_1_T1c_sus_nml.nii.gz')
T2 = ni.load_image(PATH + 'brats_2013_pat0001_1_T2_sus_nml.nii.gz')

label = io.imread('/mnt/filsystem1/code/dsb2017/code/zang/labels/pat0001_75.png')
compare = io.imread('/mnt/filsystem1/code/dsb2017/code/zang/train_set/pat0001_75.png')
temp = compare.reshape(4, 240, 240)
'''
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

n_full=[279270, 287043, 469990, 15765437, 15956157, 15773210]
n_core=[42268, 92037, 58028, 16135403, 16151163, 16185172]
n_advancing=[135487, 189613, 172181, 16016893, 16053587, 16071019]

print ' full_tumor:{}\n core_tumor:{}\n advancing_tumor:{}\n'.format(n_full, n_core, n_advancing)

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