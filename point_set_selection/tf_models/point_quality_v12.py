import numpy as np
import tensorflow as tf
import tflearn
import sys


def correlation(x):
  x=tflearn.layers.conv.conv_1d(x,192,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
  x=tf.contrib.layers.instance_norm(x)
  x=tflearn.layers.conv.conv_1d(x,64,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
  x=tf.contrib.layers.instance_norm(x)
  x=tflearn.layers.conv.conv_1d(x,2,filter_size=1,strides=1)
  x_final = x
  return x_final

def two_point_quality(single_point, gq_feat, batch_size, TOP_K, TOP_K2=1024,Nor=None):
  tmp_gq_list = []
  batch_id = tf.range(start=0,limit=batch_size)
   
  def gq_example(i):
    tmp_single_point = tf.expand_dims(single_point[i],axis=1)
    tmp_single_point = tf.tile(tmp_single_point,[1,TOP_K2,1])
    tmp_gq_feat = tf.expand_dims(gq_feat[i],axis=0)
    tmp_gq_feat = tf.tile(tmp_gq_feat,[TOP_K,1,1])
    concate_feat = tf.concat([tmp_gq_feat,tmp_single_point],axis=-1)
    tmp_gq = correlation(concate_feat)
    return tmp_gq

  tmp_re = tf.map_fn(gq_example,batch_id,dtype=tf.float32,swap_memory=True)
  tmp_re = tf.reshape(tmp_re,[batch_size,-1,2])
  return tmp_re[:,:,:2], tmp_re[:,:,:2] 
