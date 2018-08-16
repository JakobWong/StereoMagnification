import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm
import numpy as np
from scipy.misc import imsave, imresize, toimage
import os
from utils.metrics import psnr, sharp_diff, ssim
from utils.tf_utils import conv_block, deconv_block, tensor_norm, inception_model
from utils.loss import VGG_loss 
import constants as c

from skimage.transform import resize


class StereoMagificationModel:
    def __init__(self, session, summary_writer):
        """
        Initializes an appearance flow model.

        @param session: The TensorFlow Session.
        @param summary_writer: The writer object to record TensorBoard summaries
        
        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        """
        self.sess = session
        self.summary_writer = summary_writer

        self.define_graph()
    
    def define_graph(self):
        """
        Set up the model graph
        """
        with tf.name_scope('data'):
            self.ref_image = tf.placeholder(tf.float32, shape=[None,128,128,3], name='ref_image')
            self.multi_plane = tf.placeholder(tf.float32, shape=[None,128,128,3*c.NUM_PLANE])
            self.gt = tf.placeholder(tf.float32,shape=[None,128,128,3], name='gt')

        self.summaries=[]
        
        with tf.name_scope('predection'):
            def prediction(ref_image,multi_plane):
                net_in = tf.concat([ref_image,multi_plane],axis=-1)

                conv1_1 = conv_block(net_in,64,3,1)
                conv1_2 = conv_block(conv1_1,128,3,2)

                conv2_1 = conv_block(conv1_2,128,3,1)
                conv2_2 = conv_block(conv2_1,256,3,2)

                conv3_1 = conv_block(conv2_2,256,3,1)
                conv3_2 = conv_block(conv3_1,256,3,1)
                conv3_3 = conv_block(conv3_2,512,3,2)

                # weight3_1 = tf.Variable(tf.random_normal([3, 3, 512]))
                # weight3_2 = tf.Variable(tf.random_normal([3, 3, 512]))
                # weight3_3 = tf.Variable(tf.random_normal([3, 3, 512]))

                # conv4_1 = tf.nn.dilation2d(conv3_3,weight3_1,[1,1,1,1],[1,2,2,1],'SAME')
                # conv4_2 = tf.nn.dilation2d(conv4_1,weight3_2,[1,1,1,1],[1,2,2,1],'SAME')
                # conv4_3 = tf.nn.dilation2d(conv4_2,weight3_3,[1,1,1,1],[1,2,2,1],'SAME')

                conv4_1 = tf.layers.conv2d(conv3_3,512,(3,3),(1,1),'SAME',dilation_rate=(2,2))
                conv4_2 = tf.layers.conv2d(conv4_1,512,(3,3),(1,1),'SAME',dilation_rate=(2,2))
                conv4_3 = tf.layers.conv2d(conv4_2,512,(3,3),(1,1),'SAME',dilation_rate=(2,2))

                conv5_1 = deconv_block(tf.concat([conv4_3,conv3_3],axis=-1),256,4,2)
                conv5_2 = conv_block(conv5_1,256,3,1)
                conv5_3 = conv_block(conv5_2,256,3,1)

                conv6_1 = deconv_block(tf.concat([conv5_3,conv2_2],axis=-1),128,4,2)
                conv6_2 = conv_block(conv6_1,128,3,1)
                
                conv7_1 = deconv_block(tf.concat([conv6_2,conv1_2],axis=-1),64,4,2)
                conv7_2 = conv_block(conv7_1,64,3,1)
                conv7_3 = tf.layers.conv2d(conv7_2,62,(1,1),(1,1),'SAME')
                conv7_3 = tf.nn.tanh(conv7_3)

                blending_weights, alpha_images = tf.split(conv7_3,[c.NUM_PLANE,c.NUM_PLANE],axis=-1)
                blending_weights = tensor_norm(blending_weights)
                #alpha_images = tensor_norm(alpha_images)
                alpha_images = tf.nn.softmax(alpha_images,axis=-1)
               
                feature_maps = {
                    'conv1_1':conv1_1,
                    'conv1_2':conv1_2,
                    'conv2_1':conv2_1,
                    'conv2_2':conv2_2,
                    'conv3_1':conv3_1,
                    'conv3_2':conv3_2,
                    'conv3_3':conv3_3,
                    'conv4_1':conv4_1,
                    'conv4_2':conv4_2,
                    'conv4_3':conv4_3,
                    'conv5_1':conv5_1,
                    'conv6_1':conv6_1,
                    'conv6_2':conv6_2,
                    'conv7_1':conv7_1,
                    'conv7_2':conv7_2,
                    'conv7_3':conv7_3
                }

                return blending_weights, alpha_images, feature_maps
            
            
            self.blending_weights, self.alpha_images, self.feature_maps = prediction(self.ref_image,self.multi_plane)
            self.color_images = []
            for i in range(c.NUM_PLANE):
                tmp_weights = tf.expand_dims(self.blending_weights[:,:,:,i],axis=-1)
                #tmp_weights = self.blending_weights[:,:,:,i]
                self.color_images.append(
                    tf.multiply(tmp_weights,self.ref_image) + 
                    tf.multiply(1-tmp_weights,self.multi_plane[:,:,:,3*i:3*(i+1)]))
            
            self.preds = []
            for i in range(c.NUM_PLANE):
                tmp_alpha = tf.expand_dims(self.alpha_images[:,:,:,i],axis=-1)
                self.preds.append(tf.multiply(tmp_alpha, self.color_images[i]))
            self.preds = tf.accumulate_n(self.preds)
            #self.preds = inception_model(self.preds,6)

        with tf.name_scope('train'):
            self.loss = VGG_loss(self.preds,self.gt)
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE, name='optimizer')
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
            loss_summary = tf.summary.scalar('train_loss', self.loss)
            self.summaries.append(loss_summary)

        with tf.name_scope('error'):
            self.psnr = psnr(self.preds,self.gt)
            self.sharpdiff = sharp_diff(self.preds,self.gt)
            self.ssim = ssim(self.preds, self.gt)
            summary_psnr = tf.summary.scalar('train_PSNR',self.psnr)
            summary_sharpdiff = tf.summary.scalar('train_SharpDiff',self.sharpdiff)
            summary_ssim = tf.summary.scalar('trian_ssim',self.ssim)
            self.summaries += [summary_psnr, summary_sharpdiff, summary_ssim]
        self.summaries = tf.summary.merge(self.summaries)
        
    
    def train_step(self, batch, discriminator=None):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [c.BATCH_SIZE x self.height x self.width x (3 * (c.HIST_LEN + 1))].
                      The input and output frames, concatenated along the channel axis (index 3).
        @param discriminator: The discriminator model. Default = None, if not adversarial.

        @return: The global step.
        """
        ##
        # Split into inputs and outputs
        ##

        ref_image = batch['ref_image']
        multi_plane = batch['multi_plane']
        gt = batch['gt']
        batch_size = ref_image.shape[0]

        ##
        # Train
        ##

        feed_dict = {self.ref_image: ref_image, 
                     self.multi_plane: multi_plane, 
                     self.gt: gt}

        # Run the generator first to get generated images
        preds = self.sess.run(self.preds, feed_dict=feed_dict)

        _, loss, global_psnr, global_sharpdiff, global_ssim, global_step, summaries = \
            self.sess.run([self.train_op,
                           self.loss,
                           self.psnr,
                           self.sharpdiff,
                           self.ssim,
                           self.global_step,
                           self.summaries],
                           feed_dict=feed_dict)

        ##
        # User output
        ##
        if global_step % c.STATS_FREQ == 0:
            print('EXAMPLE-BASED EDSR FlowModel : Step ', global_step)
            print('                 Global Loss    : ', loss)
            print('                 PSNR           : ', global_psnr)
            print('                 Sharpdiff      : ', global_sharpdiff)
            print('                 SSIM           : ', global_ssim)

        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print('GeneratorModel: saved summaries')
        if global_step % c.IMG_SAVE_FREQ == 0:
            print('-' * 30)
            print('Saving images...')

            
            for pred_num in range(batch_size):
                pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' + str(global_step),
                                                    str(pred_num)))

                # save input images
                ref_img = ref_image[pred_num, :, :, :]
                imsave(os.path.join(pred_dir, 'ref_image.png'), imresize(ref_img,[128,128]))

                hr_img = multi_plane_img = multi_plane[pred_num, :, :, 0:3]
                imsave(os.path.join(pred_dir, 'hr_image.png'), hr_img)

                plane_dir = c.get_dir(os.path.join(pred_dir,'planes'))
                for i in range(1,c.NUM_PLANE):
                    multi_plane_img = multi_plane[pred_num, :, :, 3*i:3*i+3]
                    imsave(os.path.join(plane_dir, 'plane_%d.png'%(i+1)), multi_plane_img)

                gt_img = gt[pred_num, :, :, :]
                imsave(os.path.join(pred_dir, 'gt.png'), gt_img)

                gen_img = preds[pred_num,:,:,:]
                imsave(os.path.join(pred_dir, 'pred.png'), gen_img)

            print('Saved images!')
            print('-' * 30)

        return global_step
    
    def test_batch(self, batch, global_step, save_feature_maps=True,save_imgs=True):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [batch_size x self.height x self.width x (3 * (c.HIST_LEN+ num_rec_out))].
                      A batch of the input and output frames, concatenated along the channel axis
                      (index 3).
        @param global_step: The global step.
        @param save_imgs: Whether or not to save the input/output images to file. Default = True.

        @return: A tuple of (psnr error, sharpdiff error) for the batch.
        """

        print('-' * 30)
        print('Testing:')

        ##
        # Split into inputs and outputs
        ##

        ref_image = batch['ref_image']
        multi_plane = batch['multi_plane']
        gt = batch['gt']
        batch_size = ref_image.shape[0]

        feed_dict = {self.ref_image: ref_image, 
                     self.multi_plane: multi_plane, 
                     self.gt: gt}
        
        if save_feature_maps:
            preds, psnr, sharpdiff, ssim, feature_maps, blending_weights, alpha_images = self.sess.run([self.preds,
                                                                        self.psnr,
                                                                        self.sharpdiff,
                                                                        self.ssim,
                                                                        self.feature_maps,
                                                                        self.blending_weights,
                                                                        self.alpha_images],
                                                                        feed_dict=feed_dict)

        else:
            # Run the generator first to get generated images
            preds, psnr, sharpdiff, ssim = self.sess.run([self.preds,
                                                    self.psnr,
                                                    self.sharpdiff,
                                                    self.ssim],
                                                    feed_dict=feed_dict)
        ##
        # User output
        ##

        print('PSNR      : ', psnr)
        print('Sharpdiff : ', sharpdiff)
        print('SSIM      : ', ssim)

        print('-' * 30)
        print('Saving images...')
        for pred_num in range(batch_size):
            pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Test/Step_' + str(global_step),
                                                str(pred_num)))

            # save input images
            ref_img = ref_image[pred_num, :, :, :]
            imsave(os.path.join(pred_dir, 'ref_image.png'), imresize(ref_img,[128,128]))

            hr_img = multi_plane_img = multi_plane[pred_num, :, :, 0:3]
            imsave(os.path.join(pred_dir, 'hr_image.png'), hr_img)

            plane_dir = c.get_dir(os.path.join(pred_dir,'planes'))
            for i in range(1,c.NUM_PLANE):
                multi_plane_img = multi_plane[pred_num, :, :, 3*i:3*(i+1)]
                imsave(os.path.join(plane_dir, 'plane_%d.png'%(i+1)), multi_plane_img)

            gt_img = gt[pred_num, :, :, :]
            imsave(os.path.join(pred_dir, 'gt.png'), gt_img)

            gen_img = preds[pred_num,:,:,:]
            imsave(os.path.join(pred_dir, 'pred.png'), gen_img)
            
            if save_feature_maps:
                feature_dir = c.get_dir(os.path.join(pred_dir,'features'))
                for layer in feature_maps:
                    layer_dir = c.get_dir(os.path.join(feature_dir,layer))
                    layer_img = feature_maps[layer][pred_num,:,:,:]
                    num_feature = (layer_img.shape)[2]
                    for k in range(num_feature):
                        imsave(os.path.join(layer_dir, '%d.bmp'%k), layer_img[:,:,k])

                blending_weights_dir = c.get_dir(os.path.join(feature_dir,'blending_weights'))
                alpha_dir = c.get_dir(os.path.join(feature_dir,'alpha'))
                for k in range(c.NUM_PLANE):
                    imsave(os.path.join(blending_weights_dir, '%d.bmp'%k), blending_weights[pred_num,:,:,k])
                    imsave(os.path.join(alpha_dir, '%d.bmp'%k), alpha_images[pred_num,:,:,k])
        print('Saved images!')
        print('-' * 30)
