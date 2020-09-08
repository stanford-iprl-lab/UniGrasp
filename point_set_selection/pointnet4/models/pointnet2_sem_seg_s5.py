import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
print("BASE_DIR",BASE_DIR)
import tensorflow as tf
import numpy as np
from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util
import tflearn

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None,gripper_feat=None,env_feat=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.01, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.02, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.04, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=64, radius=0.08, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=48, radius=0.16, nsample=32, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5')
    l6_xyz, l6_points, l6_indices = pointnet_sa_module(l5_xyz, l5_points, npoint=4, radius=0.20, nsample=32, mlp=[1024,1024,2048], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer6')


    if env_feat is None:
      extra_feat = gripper_feat
    else:
      extra_feat = tf.concat([gripper_feat,env_feat],axis=-1)

    extra_feat = tf.expand_dims(extra_feat,axis=1)
    extra_feat0 = extra_feat
 
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,512,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,256,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,256,filter_size=1,strides=1,activation=tf.nn.leaky_relu)

    extra_feat2 = extra_feat
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,128,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,128,filter_size=1,strides=1,activation=tf.nn.leaky_relu)
    extra_feat = tflearn.layers.conv.conv_1d(extra_feat,64,filter_size=1,strides=1,activation=tf.nn.leaky_relu)

    extra_feat5 = extra_feat

    extra_feat0 = tf.tile(extra_feat0,[1,4,1])
    l6_points = tf.concat([l6_points,extra_feat0],axis=-1)

    # Feature Propagation layers
    l5_points = pointnet_fp_module(l5_xyz, l6_xyz, l5_points, l6_points, [2048,2048,1024], is_training, bn_decay, scope='fa_layer5',bn=True)
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [1024,1024,512], is_training, bn_decay, scope='fa_layer0',bn=True)
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512,384], is_training, bn_decay, scope='fa_layer1',bn=True)
 
    extra_feat2 = tf.tile(extra_feat2,[1,128,1])
    l3_points = tf.concat([l3_points,extra_feat2],axis=-1)

    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [384,384,256], is_training, bn_decay, scope='fa_layer2',bn=True)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,256,128], is_training, bn_decay, scope='fa_layer3',bn=True)
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,64], is_training, bn_decay, scope='fa_layer4',bn=True)
 
    extra_feat5 = tf.tile(extra_feat5,[1,2048,1])
    l0_points = tf.concat([l0_points,extra_feat5],axis=-1)
    l0_points = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1_3', bn_decay=bn_decay)
    l0_points = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1_4', bn_decay=bn_decay)
 
    net = l0_points #tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 

    return end_points

def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
