import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import constants as c
import os
##
# Data
##

def get_train_batch(indices):
    """
    Loads c.BATCH_SIZE training images from the database.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """

    ref_image = np.empty([len(indices), c.HR_HEIGHT, c.HR_WIDTH, 3],dtype=np.float32)
    multi_plane = np.empty([len(indices), c.HR_HEIGHT, c.HR_WIDTH, 3*c.NUM_PLANE],dtype=np.float32)
    gt = np.empty([len(indices), c.HR_HEIGHT, c.HR_WIDTH, 3],dtype=np.float32)
    for i in range(len(indices)):
        idx = str(indices[i]+1).zfill(5)
        lr_path = c.TRAIN_DIR + '/lr/' + idx + '.png'
        ref_image[i,:,:,:] = imresize(imread(lr_path),(128,128))/255.0

        psv_directory = os.path.join(c.TRAIN_DIR,'psv',idx)
        for j in range(c.NUM_PLANE):
            psv_path = os.path.join(psv_directory,str(j+1).zfill(2)+'.png') 
            multi_plane[i,:,:,3*j:3*(j+1)] = imread(psv_path)/255.0
        
        gt_path = c.TRAIN_DIR + '/gt/' + idx + '.png'
        gt[i,:,:,:] = imread(gt_path)/255.0

    res = {'ref_image':ref_image,'multi_plane':multi_plane,'gt':gt}
    return res

def get_test_batch():
    """
    Loads c.BATCH_SIZE training images from the database.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    ref_image = np.empty([c.TEST_BATCH_SIZE, c.HR_HEIGHT, c.HR_WIDTH, 3],dtype=np.float32)
    multi_plane = np.empty([c.TEST_BATCH_SIZE, c.HR_HEIGHT, c.HR_WIDTH, 3*c.NUM_PLANE],dtype=np.float32) 
    gt = np.empty([c.TEST_BATCH_SIZE, c.HR_HEIGHT, c.HR_WIDTH, 3],dtype=np.float32)
    for i in range(c.TEST_BATCH_SIZE):
        #idx = str(np.random.randint(1,c.NUM_TEST_SAMPLE+1)).zfill(5)
        idx = str(i+1).zfill(5)
        lr_path = c.TEST_DIR + '/lr/' + idx + '.png'
        ref_image[i,:,:,:] = imresize(imread(lr_path),(128,128))/255.0
        
        psv_directory = os.path.join(c.TEST_DIR,'psv',idx)
        for j in range(c.NUM_PLANE):
            psv_path = os.path.join(psv_directory,str(j+1).zfill(2)+'.png')  
            multi_plane[i,:,:,3*j:3*(j+1)] = imread(psv_path)/255.0
        
        gt_path = c.TEST_DIR + '/gt/' + idx + '.png'
        gt[i,:,:,:] = imread(gt_path)/255.0

    res = {'ref_image':ref_image,'multi_plane':multi_plane,'gt':gt}

    return res
