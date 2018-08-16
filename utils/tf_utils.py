import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm

def conv_block(inputs,output_channels,kernel_size,stride):
    preds = tf.layers.conv2d(inputs,
                            output_channels,
                            (kernel_size,kernel_size),
                            (stride,stride),
                            'SAME')
    preds = tf.nn.relu(preds)
    preds = layer_norm(preds)
    return preds

def deconv_block(inputs,output_channels,kernel_size,stride):
    preds = tf.layers.conv2d_transpose(inputs,
                            output_channels,
                            (kernel_size,kernel_size),
                            (stride,stride),
                            'SAME')
    preds = tf.nn.relu(preds)
    preds = layer_norm(preds)
    return preds

def tensor_norm(tensor):
    tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), 
                    tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
    return tensor

def inception_module(inputs):
    feature1 = tf.layers.conv2d(inputs,64,[3,3],[1,1], padding='SAME')
    #print(feature1)
    feature1 = tf.layers.batch_normalization(feature1)
    #print(feature1)

    feature2 = tf.layers.conv2d(inputs,32,[5,5],[1,1], padding='SAME')
    #print(feature2)
    feature2 = tf.layers.batch_normalization(feature2)
    #print(feature2)

    residual = tf.concat([feature1,feature2],3)

    output = tf.add(inputs,residual)
    #print(output)
    relu_output = tf.nn.leaky_relu(output,0.25)
    #print(relu_output)

    return relu_output

def inception_model(inputs,n_module):
    feature1 = tf.layers.conv2d(inputs,64,[3,3],[1,1], padding='SAME')
    #print(feature1)
    feature1 = tf.layers.batch_normalization(feature1)
    #print(feature1)

    feature2 = tf.layers.conv2d(inputs,32,[5,5],[1,1], padding='SAME')
    #print(feature2)
    feature2 = tf.layers.batch_normalization(feature2)
    #print(feature2)

    feature_low = tf.concat([feature1,feature2],3)
    feature_low_relu = tf.nn.leaky_relu(feature_low,0.25)
    #print(feature_low_relu)

    feature = feature_low_relu
    for i in range(n_module):
        with tf.name_scope('inception_modules_{}'.format(i)):
            feature = inception_module(feature)

    feature_high = tf.layers.conv2d(feature,96,[5,5],[1,1], padding='SAME')
    feature_high = tf.layers.batch_normalization(feature_high)
    feature_high_relu = tf.nn.leaky_relu(feature_high,0.25)
    #print(feature_high_relu)

    feature_all = tf.concat([feature_low_relu,feature_high_relu],3)
    #print(feature_all)

    nc_out = inputs.get_shape().as_list()[3]
    outputs = tf.layers.conv2d(feature_all,nc_out,[5,5],[1,1], padding='SAME',activation=tf.tanh)
    #print(outputs)

    return outputs