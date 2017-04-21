import os
import numpy as np
from skimage import io
from joblib import Parallel, delayed

ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/zang/'
BRATS_PATH = ORI_PATH + 'brats_val/'
SAVE_PATH = ORI_PATH + 'val_set/'
LABEL_PATH = ORI_PATH + 'labels/'

def generate(folders):

	patient = folders.split('_')[2]
	PATIENT_PATH = BRATS_PATH + folders + '/'
	OT_PATH = PATIENT_PATH + 'brats_2013_{}_1_OT/'.format(patient)
	FLAIR_PATH = PATIENT_PATH + 'brats_2013_{}_1_Flair_sus_nml/'.format(patient)
	T1_PATH = PATIENT_PATH + 'brats_2013_{}_1_T1_sus_nml/'.format(patient)
	T1C_PATH = PATIENT_PATH + 'brats_2013_{}_1_T1c_sus_nml/'.format(patient)
	T2_PATH = PATIENT_PATH + 'brats_2013_{}_1_T2_sus_nml/'.format(patient)

	for label in os.listdir(OT_PATH):

		label_id = label.split('.')[0]
		label_id = label_id.split('_')[5]

		label_img = io.imread(OT_PATH+label)
		io.imsave(LABEL_PATH + '{}_{}.png'.format(patient, label_id), label_img)
		T1c = io.imread(T1C_PATH + 'brats_2013_{}_1_T1c_sus_nml_{}.png'.format(patient, label_id))
		img_np = np.asarray(T1c)
		patch_save = np.zeros((4,240,240))

		if np.max(img_np) > 0:

			Flair = io.imread(FLAIR_PATH + 'brats_2013_{}_1_Flair_sus_nml_{}.png'.format(patient, label_id))
			T1 = io.imread(T1_PATH + 'brats_2013_{}_1_T1_sus_nml_{}.png'.format(patient, label_id))
			T2 = io.imread(T2_PATH + 'brats_2013_{}_1_T2_sus_nml_{}.png'.format(patient, label_id))
			temp = [Flair, T1, T1c, T2]

			for channel in xrange(len(patch_save)):
				patch_save[channel] = np.asarray(temp[channel], 'float32')

			patch_save = patch_save.reshape(960,240)

			if np.max(patch_save) != 0:
				patch_save /= np.max(patch_save)
			if np.min(patch_save) <= -1:
				patch_save /= abs(np.min(patch_save))

			io.imsave(SAVE_PATH + '{}_{}.png'.format(patient, label_id), patch_save)
		else:
			continue
	print 'finish saving patient:{}'.format(patient)


if __name__ == '__main__':

	Parallel(n_jobs=12)(delayed(generate)(folders) for folders in os.listdir(BRATS_PATH))



