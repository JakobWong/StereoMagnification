import numpy as np
import os
from glob import glob
import shutil
from datetime import datetime
from scipy.ndimage import imread

##
# Data
##

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '_').replace(':', '.')[:-10]

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def get_test_frame_dims():
    img_path = glob(os.path.join(TEST_DIR, '*/*'))[0]
    print(img_path)
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_train_frame_dims():
    img_path = glob(os.path.join(TRAIN_DIR, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def set_test_dir(directory):
    """
    Edits all constants dependent on TEST_DIR.

    @param directory: The new test directory.
    """
    global TEST_DIR, FULL_HEIGHT, FULL_WIDTH

    TEST_DIR = directory
    FULL_HEIGHT, FULL_WIDTH = get_test_frame_dims()

# root directory for all data
DATA_DIR = get_dir('/home/server/Desktop/jh/data')
# directory of unprocessed training frames
TRAIN_DIR = os.path.join(DATA_DIR, 'autumn_psv_5000')
# directory of unprocessed test frames
TEST_DIR = os.path.join(DATA_DIR, 'autumn_psv_20000')
# Directory of processed training clips.
# hidden so finder doesn't freeze w/ so many files. DON'T USE `ls` COMMAND ON THIS DIR!
# TRAIN_DIR_CLIPS = get_dir(os.path.join(DATA_DIR, '.Clips/'))

# For processing clips. l2 diff between frames must be greater than this
MOVEMENT_THRESHOLD = 100
# total number of processed clips in TRAIN_DIR_CLIPS
# NUM_CLIPS = len(glob(TRAIN_DIR_CLIPS + '*'))
NUM_TRAIN_SAMPLE = 5000
NUM_TEST_SAMPLE = 20000

# the height and width of the full frames to test on. Set in avg_runner.py or process_data.py main.
FULL_HEIGHT = 128
FULL_WIDTH = 128
# the height and width of the LR inputs to train on
LR_HEIGHT = LR_WIDTH = 32
# the height and width of the HR inputs to train on
HR_HEIGHT = HR_WIDTH = 128

##
# Output
##

def set_save_name(name):
    """
    Edits all constants dependent on SAVE_NAME.

    @param name: The new save name.
    """
    global SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, IMG_SAVE_DIR

    SAVE_NAME = name
    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
    IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', SAVE_NAME))

def clear_save_name():
    """
    Clears all saved content for SAVE_NAME
    """
    clear_dir(MODEL_SAVE_DIR)
    clear_dir(SUMMARY_SAVE_DIR)
    clear_dir(IMG_SAVE_DIR)


# root directory for all saved content
SAVE_DIR = get_dir('../Save_lr_as_bkg(soft_max_alpha)/Aug_09/')

# inner directory to differentiate between runs
SAVE_NAME = 'Default/'
# directory for saved models
MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
# directory for saved images
IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', SAVE_NAME))


STATS_FREQ      = 10     # how often to print loss/train error stats, in # steps
SUMMARY_FREQ    = 500    # how often to save the summaries, in # steps
IMG_SAVE_FREQ   = 200   # how often to save generated images, in # steps
TEST_FREQ       = 200   # how often to test the model on test data, in # steps
MODEL_SAVE_FREQ = 500  # how often to save the model, in # steps

##
# General training
##

# the training minibatch size
TRAIN_BATCH_SIZE = 15
# the testing minibatch size
TEST_BATCH_SIZE = 20
# the number of history frames to give as input to the network
HIST_LEN = 2

# learning rate for the generator model
LRATE = 0.0002  # Value in paper is 0.04

LRATE_D = 0.04
ADVERSARIAL = False

CONV_FMS_D = [3, 128, 256, 512, 128]
KERNEL_SIZES_D = [9, 9, 7, 7]
FC_LAYER_SIZES_D = [1024, 512, 1]
DROP_OUT_RATES = [0.5,0.5,0.5]

PADDING_D = 'VALID'
NUM_PLANE=31
