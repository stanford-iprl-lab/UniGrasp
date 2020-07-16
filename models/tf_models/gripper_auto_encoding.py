import numpy as  np
import tensorflow as tf
import tflearn
import sys
import os

def pc_encoder_v1(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'encode'):
      pc_feat = tflearn.layers.conv.conv_1d(inputs,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.max_pool_1d(pc_feat,2048,strides=2048,padding='valid')
      return pc_feat       

def pc_encoder(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'encode'):
      pc_feat = tflearn.layers.conv.conv_1d(inputs,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,256,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,256,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.max_pool_1d(pc_feat,2048,strides=2048,padding='valid')
      return pc_feat       

def pc_encoder_v3(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'encode'):
      pc_feat = tflearn.layers.conv.conv_1d(inputs,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,256,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,512,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.max_pool_1d(pc_feat,2048,strides=2048,padding='valid')
      return pc_feat       


def pc_decoder(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'decode'):
      pc_feat = tflearn.layers.core.fully_connected(inputs,512,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.core.fully_connected(pc_feat,1024,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc      = tflearn.layers.core.fully_connected(pc_feat,2048 * 3,activation='linear',weight_decay=1e-3,regularizer='L2')
      pc = tf.reshape(pc,(-1,2048,3))
      return pc 

w_init = tflearn.initializations.truncated_normal(shape=(1024,2048 * 3),stddev=0.1)
w_init_1 = tflearn.initializations.truncated_normal(shape=(512, 1024),stddev=0.1)
def pc_decoder_v2(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'decode'):
      pc_feat = tflearn.layers.core.fully_connected(inputs,512,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.core.fully_connected(pc_feat,1024,activation='relu',weights_init=w_init_1,weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc      = tflearn.layers.core.fully_connected(pc_feat,2048 * 3,activation='linear',weights_init=w_init, weight_decay=1e-3,regularizer='L2')
      pc = tf.reshape(pc,(-1,2048,3))
      return pc 

def gripper_model(in_pc):
  pc_feat = pc_encoder(in_pc,scope='pc_encoder')
  out_pc = pc_decoder(pc_feat,scope='pc_decoder') 
  return out_pc, pc_feat
