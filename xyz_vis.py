import os
import numpy as np
from skimage import io
import nipy as ni
import matplotlib.pyplot as plt

PATH = '/mnt/filsystem1/code/dsb2017/code/zang/predict/'
FOLDER_LIST = ['25', '13', '14', '06']
architect_list = ['triple', 'double', 'local', 'global']

def process(patch_save):
	if np.max(patch_save) != 0:
		patch_save /= np.max(patch_save)
	if np.min(patch_save) <= -1:
		patch_save /= abs(np.min(patch_save))
	return patch_save

cube=np.zeros((240,240,240,3))

for f in range(4):
	for a in range(4):
		FOLDER = FOLDER_LIST[f]
		architect = architect_list[a]

		for i in range(240):
			if os.path.exists(PATH+'{}_seg/pat00{}_{}.png'.format(architect, FOLDER, i)):
				img = io.imread(PATH+'{}_seg/pat00{}_{}.png'.format(architect, FOLDER, i))
				#im_slice = np.asarray(img)
				cube[i,:,:,:] = img
			else: 
				pass
		print FOLDER, architect
		np.save(PATH+'label_{}'.format(FOLDER),cube)

		cube = np.load(PATH+'label_{}.npy'.format(FOLDER))

		if not os.path.exists(PATH+'{}/'.format(architect)):
			os.mkdir(PATH+'{}/'.format(architect))
		if not os.path.exists(PATH+'{}/pat{}/'.format(architect, FOLDER)):
			os.mkdir(PATH+'{}/pat{}/'.format(architect, FOLDER))

		if not os.path.exists(PATH+'{}/pat{}/pat_xy/'.format(architect, FOLDER)):
			os.mkdir(PATH+'{}/pat{}/pat_xy/'.format(architect, FOLDER))
		if not os.path.exists(PATH+'{}/pat{}/pat_yz/'.format(architect, FOLDER)):
			os.mkdir(PATH+'{}/pat{}/pat_yz/'.format(architect, FOLDER))
		if not os.path.exists(PATH+'{}/pat{}/pat_xz/'.format(architect, FOLDER)):
			os.mkdir(PATH+'{}/pat{}/pat_xz/'.format(architect, FOLDER))

		for i in range(240):
			img_xy = cube[i,:,:,:]
			img_xy = process(img_xy)
			io.imsave(PATH+'{}/pat{}/pat_xy/pat00{}_{}.png'.format(architect, FOLDER, FOLDER, i), img_xy)
		for i in range(240):
			img_yz = cube[:,i,:,:]
			img_yz = process(img_yz)
			io.imsave(PATH+'{}/pat{}/pat_yz/pat00{}_{}.png'.format(architect, FOLDER, FOLDER, i), img_yz)
		for i in range(240):
			img_xz = cube[:,:,i,:]
			img_xz = process(img_xz)
			io.imsave(PATH+'{}/pat{}/pat_xz/pat00{}_{}.png'.format(architect, FOLDER, FOLDER, i), img_xz)

# example = io.imread(PATH+'seg/pat0014_70.png')

# plt.subplot(221)
# plt.imshow(cube[70,:,:,:])
# plt.subplot(222)
# plt.imshow(cube[:,120,:,:])
# plt.subplot(223)
# plt.imshow(cube[:,:,120,:])
# plt.subplot(224)
# plt.imshow(example)
# plt.show()