# -*- coding:utf -*-
import numpy as np
import subprocess
import random
import progressbar
from glob import glob
from skimage import io
import nipy as ni

np.random.seed(5) # for reproducibility 使每次产生的随机数不变
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
# A text progress bar is typically used to display the progress of a long running operation, providing a visual cue that processing is underway.

ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/zang/'

class BrainPipeline(object):
    '''
    A class for processing brain scans for one patient
    INPUT:  (1) filepath 'path': path to directory of one patient. Contains following mha files:
            flair, t1, t1c, t2, ground truth (gt)
            (2) bool 'n4itk': True to use n4itk normed t1 scans (defaults to True)
            (3) bool 'n4itk_apply': True to apply and save n4itk filter to t1 and t1c scans for given patient. This will only work if the
    '''
    def __init__(self, path, n4itk = True, n4itk_apply = False):
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        
        self.slices_by_mode, n = self.read_scans()

        self.slices_by_slice = n
        self.normed_slices = self.norm_slices()

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans.
        transforms scans of same slice into strip of 5 images
        '''
        print 'Loading scans...'
        slices_by_mode = np.zeros((5, 155, 240, 240))   #变量初始化
        slices_by_slice = np.zeros((155, 5, 240, 240))
        b=np.zeros((155,240,240))

        flair = glob(self.path + '/*Flair*.nii.gz')
        t2 = glob(self.path + '/*_T2*.nii.gz')
        gt = glob(self.path + '/*OT.nii.gz')
        t1s = glob(self.path + '/*_T1_*.nii.gz')   # T1
        t1_n4 = glob(self.path + '/*_T1c_*.nii.gz') # T1c 

        t1 = [scan for scan in t1s if scan not in t1_n4]
        scans = [flair[0], t1s[0], t1_n4[0], t2[0], gt[0]] # directories to each image (5 total)

        if self.n4itk_apply:
            print '-> Applyling bias correction...'
            for t1_path in t1:
                self.n4itk_norm(t1_path) # normalize files
                scans = [flair[0], t1_n4[0], t1_n4[0], t2[0], gt[0]]
        elif self.n4itk:
            scans = [flair[0], t1_n4[0], t1_n4[0], t2[0], gt[0]]
            print "=================="
        for scan_idx in xrange(5):
            s= ni.load_image(flair[0])
    	    for i in range(155):
                b[i,:,:]=np.asarray(s[:,:,i])
                # read each image directory, save to self.slices
                slices_by_mode[scan_idx] = b

            for mode_ix in xrange(slices_by_mode.shape[0]): # modes 1 thru 5
                for slice_ix in xrange(slices_by_mode.shape[1]): # slices 1 thru 155
                    slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix] # reshape by slice

        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        '''
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        if n4itk == True, will apply n4itk bias correction to T1 and T1c images
        '''
        print 'Normalizing slices...'
        normed_slices = np.zeros((155, 5, 240, 240))
        for slice_ix in xrange(155):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] =  self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print 'Done.'
        return normed_slices

    def _normalize(self, slice):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
                (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (0.5,99.5))  # np.percentile  百分位
        slice = np.clip(slice, b, t)  # b: slice_min, t:slice_max 
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)    #标准化slice

    def save_patient(self, reg_norm_n4, patient_num):
        '''
        INPUT:  (1) int 'patient_num': unique identifier for each patient
                (2) string 'reg_norm_n4': 'reg' for original images, 'norm' normalized images, 'n4' for n4 normalized images
        OUTPUT: saves png in Norm_PNG directory for normed, Training_PNG for reg
        '''
        print 'Saving scans for patient {}...'.format(patient_num)
        progress.currval = 0
        if reg_norm_n4 == 'norm': #saved normed slices
            for slice_ix in progress(xrange(155)): # reshape to strip
                strip = self.normed_slices[slice_ix].reshape(1200, 240)

                if np.max(strip) != 0: # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1: # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
		#print 'the max of strip:',np.max(strip)
		#print "the min of strip:",np.min(strip)
                io.imsave(ORI_PATH+'Norm_PNG/{}_{}.jpg'.format(patient_num, slice_ix), strip)

        elif reg_norm_n4 == 'reg':
            for slice_ix in progress(xrange(155)):
                strip = self.slices_by_slice[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                io.imsave(ORI_PATH+'Training_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

        else:
            for slice_ix in progress(xrange(155)): # reshape to strip
                strip = self.normed_slices[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0: # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1: # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
                io.imsave(ORI_PATH+'n4_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
	print 'save'

    def n4itk_norm(self, path, n_dims=3, n_iters='[20,20,10,5]'):
        '''
        INPUT:  (1) filepath 'path': path to mha T1 or T1c file
                (2) directory 'parent_dir': parent directory to mha file
        OUTPUT: writes n4itk normalized image to parent_dir under orig_filename_n.mha
        '''
        output_fn = path[:-4] + '_n.nii'
        # run n4_bias_correction.py path n_dim n_iters output_fn
        subprocess.call('python n4_bias_correction.py ' + path + ' ' + str(n_dims) + ' ' + n_iters + ' ' + output_fn, shell = True)


def save_patient_slices(patients, type):
    '''
    INPUT   (1) list 'patients': paths to any directories of patients to save. for example- glob("Training/HGG/**")
            (2) string 'type': options = reg (non-normalized), norm (normalized, but no bias correction), n4 (bias corrected and normalized)
    saves strips of patient slices to approriate directory (Training_PNG/, Norm_PNG/ or n4_PNG/) as patient-num_slice-num
    '''
    for patient_num, path in enumerate(patients):  #path 为list 类型
        print path
        a = BrainPipeline(path)
        a.save_patient(type, patient_num)

def s3_dump(directory, bucket):
    '''
    dump files from a given directory to an s3 bucket
    INPUT   (1) string 'directory': directory containing files to save
            (2) string 'bucket': name od s3 bucket to dump files
    '''
    subprocess.call('aws s3 cp' + ' ' + directory + ' ' + 's3://' + bucket + ' ' + '--recursive')

def save_labels(fns):
    '''
    INPUT list 'fns': filepaths to all labels
    '''
    progress.currval = 0 
    slices=np.zeros((240,240)) 
    
    label=glob(fns+'/*OT.nii.gz')

    print 'len of label:',len(label)
    print 'type of label:',type(label)
        
    s =  ni.load_image(label[0])
    print s.shape
    print "=========="
    label_idx=0
    for slice_idx in xrange(1):
        slices=np.asarray(s[:,:,slice_idx])
        print slices.shape
        io.imsave(ORI_PATH+'Labels/{}_{}L.png'.format(label_idx, slice_idx), slices)
        


if __name__ == '__main__':

    save_labels(ORI_PATH+'brats_2013/')

    patients = glob(ORI_PATH+'brats_2013/**')
    print('patients:'),patients

    save_patient_slices(patients, 'reg')
    save_patient_slices(patients, 'norm')
    save_patient_slices(patients, 'n4')
    s3_dump('Graveyard/Training_PNG/', 'orig-training-png')
