import tensorflow as tf
import numpy as np
import constants as c
import vgg19.vgg19 as vgg19

def VGG_loss(y_pred,y_true):
    vgg = vgg19.Vgg19()

    vgg.build(y_pred)
    vgg_pred = {}
    vgg_pred['conv1_2'] = vgg.conv1_2
    vgg_pred['conv2_2'] = vgg.conv2_2
    vgg_pred['conv3_2'] = vgg.conv3_2
    vgg_pred['conv4_2'] = vgg.conv4_2
    vgg_pred['conv5_2'] = vgg.conv5_2

    vgg = vgg19.Vgg19() 
    vgg.build(y_true)
    vgg_true = {}
    vgg_true['conv1_2'] = vgg.conv1_2
    vgg_true['conv2_2'] = vgg.conv2_2
    vgg_true['conv3_2'] = vgg.conv3_2
    vgg_true['conv4_2'] = vgg.conv4_2
    vgg_true['conv5_2'] = vgg.conv5_2

    loss = []
    #weights = [1.0/64.0,1.0/128.0,1.0/256.0,1.0/512.0,1.0/512.0]
    weights = [1.0/128.0,1.0/64.0,1.0/32.0,1.0/16.0,1.0]
    layers = ['conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
    for i in range(5):
        layer = layers[i]
        loss.append(weights[i] * tf.reduce_mean(tf.abs(vgg_pred[layer]-vgg_true[layer])))
    loss = tf.reduce_mean(loss)
    return loss

def mae(y_pred,y_true):
    return tf.mean(tf.abs(y_pred,y_true))


