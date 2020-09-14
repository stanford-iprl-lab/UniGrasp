from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

from sklearn.neighbors import KDTree
import math
import time
import tensorflow as tf
import numpy as np
import os
import time
import sys
import argparse

from pgm_loader import read_pgm_xyz
from tf_models.point_quality_v6 import two_point_quality
from tf_models.point_quality_v12 import two_point_quality as two_point_quality_stage3

import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET2_DIR = os.path.join(BASE_DIR,'pointnet4','models')
sys.path.insert(0,POINTNET2_DIR)

from pointnet2_sem_seg import get_model as obj_model
from pointnet2_sem_seg_s4 import get_model as obj_model_v2
from pointnet2_sem_seg_s6 import get_model as obj_model_v3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR,'../')
D3VIS_DIR = os.path.join(ROOT_DIR,'vis_3d')
sys.path.insert(0,D3VIS_DIR)
from show3d_balls import showpoints

ROOT_DIR = os.path.join(BASE_DIR,'../')
GRIPPER_TOP_DIR = os.path.join(ROOT_DIR,'data/gripper_features/Data_DB')
DATA_TOP_DIR = os.path.join(ROOT_DIR,'data/real_world')

#from data_preparing_test import train_val_test_list
#print("train num %d , val num %d , test num %d" % (len(train_val_test_list._train),len(train_val_test_list._val),len(train_val_test_list._test)))

nnn = 1
# Basic model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--saver_dir',default='./saved_models/',help='Directory to save the trained model')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='Initial learning rate')
parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
parser.add_argument('--batch_size',type=int,default=nnn * 1,help='Number of examples within a batch')
parser.add_argument('--max_model_to_keep',type=int,default=400,help='max saved models')
parser.add_argument('--log_dir',default='./logging/',help='folder to save logging info')
FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.saver_dir):
  os.mkdir(FLAGS.saver_dir)

if not os.path.exists(FLAGS.log_dir):
  os.mkdir(FLAGS.log_dir)

                          
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
TOP_K = 1024
TOP_K2 = 1024

in_gripper_min_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_min')
in_gripper_max_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_max')
in_gripper_mean_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_mean')

gt_gripper_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_gt')

gripper_feat_tf = tf.placeholder(tf.float32,[None,256 * 3])


obj_pc_tf = tf.placeholder(tf.float32,[None,2048,3],'obj_pc')

rand_stage1_tf = tf.placeholder(tf.float32,[None, 2048],'rand_n')

#### stage one
with tf.variable_scope('stage1'):
  with tf.variable_scope('pointnet2'):
    gq_prediction, pred_pcn_tf, nets_end, middle_feat_tf , out_gripper_feat_tf= obj_model_v2(obj_pc_tf,is_training=True,num_class=2,bn_decay=None,gripper_feat=gripper_feat_tf,env_feat=None)
    out_single_point_feat_raw_tf = nets_end['feats']

  gt_pcn_tf = tf.placeholder(tf.float32,[None,2048,3],'gt_normal')
  single_loss_nor = tf.norm(pred_pcn_tf - gt_pcn_tf,axis=-1,keep_dims=True)
  single_loss_nor = tf.stop_gradient(single_loss_nor)
  print("single_loss_nor",single_loss_nor)
  
  loss_nor = tf.reduce_mean(-tf.reduce_sum(pred_pcn_tf * gt_pcn_tf,axis=-1,keep_dims=True)) * 100.0
  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)

  out_single_point_feat_tf = out_single_point_feat_raw_tf
  
  pred_label_tf = tf.nn.softmax(gq_prediction)[:,:,1] 

  gq_label_tf = tf.placeholder(tf.float32,[None, 2048],'gt_label')
  gq_label_weight_tf = tf.placeholder(tf.float32,[None,2048],'gt_label_weight')

  out_single_point_logits_tf = gq_prediction

  out_single_point_score_tf = tf.nn.softmax(out_single_point_logits_tf)[:,:,1]

  if 0:
    out_single_point_score_tf = out_single_point_score_tf + rand_stage1_tf

  out_single_point_top_value_tf, out_single_point_top_index_tf = tf.nn.top_k(out_single_point_score_tf,k=TOP_K,sorted=True)
 
  _, out_single_point_top_1024_index_tf = tf.nn.top_k(out_single_point_score_tf,k=TOP_K2,sorted=True)

  out_single_point_score_tf_ = out_single_point_score_tf
  denominator_stage1 = tf.reduce_sum(tf.exp(out_single_point_score_tf_),axis=1,keep_dims=True)
  loss_ListNet_Loss_s1 = -tf.reduce_mean( tf.cast(gq_label_tf,dtype=tf.float32) * gq_label_weight_tf * tf.log( tf.exp(out_single_point_score_tf_) / denominator_stage1 ) )

  loss_stage1_ce =tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gq_label_tf,dtype=tf.int32),logits=gq_prediction))

  loss_stage1 = loss_stage1_ce
  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage1)

### stage two
with tf.variable_scope('stage2'):
  gripper_feat_stage2_tf = gripper_feat_tf #gripper_model_stage2(gripper_min_tf,gripper_max_tf)

  with tf.variable_scope('pointnet2'):
    nets_end = obj_model_v3(obj_pc_tf,is_training=True,num_class=2,bn_decay=None,gripper_feat=gripper_feat_stage2_tf,env_feat=None) 
    out_single_point_feat_stage2_tf = nets_end['feats']

  pred_pcn_tf_stage2 = tf.stop_gradient(pred_pcn_tf)
  top_k_feat_list = []
  top_1024_feat_list = []
  top_1024_xyz_list = []
  top_k_xyz_list = []
  top_1024_nor_stage2_list = []
  top_k_nor_stage2_list = []

  for bi in range(FLAGS.batch_size):
    top_1024_feat_list.append(tf.gather(out_single_point_feat_stage2_tf[bi], out_single_point_top_1024_index_tf[bi]))
    top_k_feat_list.append(tf.gather(out_single_point_feat_stage2_tf[bi],out_single_point_top_index_tf[bi]))
    top_1024_xyz_list.append(tf.gather(obj_pc_tf[bi],out_single_point_top_1024_index_tf[bi]))
    top_k_xyz_list.append(tf.gather(obj_pc_tf[bi],out_single_point_top_index_tf[bi]))
    top_1024_nor_stage2_list.append(tf.gather(pred_pcn_tf_stage2[bi],out_single_point_top_1024_index_tf[bi]))
    top_k_nor_stage2_list.append(tf.gather(pred_pcn_tf_stage2[bi],out_single_point_top_index_tf[bi]))

  top_1024_feat_tf = tf.convert_to_tensor(top_1024_feat_list)
  top_k_feat_tf = tf.convert_to_tensor(top_k_feat_list)

  top_1024_xyz_tf = tf.convert_to_tensor(top_1024_xyz_list)
  top_k_xyz_tf = tf.convert_to_tensor(top_k_xyz_list)  

  top_1024_nor_stage2_tf = tf.convert_to_tensor(top_1024_nor_stage2_list)
  top_k_nor_stage2_tf = tf.convert_to_tensor(top_k_nor_stage2_list)  

  top_1024_nor_stage2_tf = tf.expand_dims(top_1024_nor_stage2_tf,axis=1)
  top_k_nor_stage2_tf = tf.expand_dims(top_k_nor_stage2_tf,axis=2)

  top_1024_nor_stage2_tf = tf.tile(top_1024_nor_stage2_tf,[1,TOP_K,1,1])
  top_k_nor_stage2_tf = tf.tile(top_k_nor_stage2_tf,[1,1,TOP_K2,1])

  out_nor_stage2_tf = tf.concat([top_k_nor_stage2_tf, top_1024_nor_stage2_tf],axis=-1)

  #out_nor_stage2_tf = tf.reshape(out_nor_stage2_tf,[FLAGS.batch_size, -1, 6])
  stage2_check_tf = tf.norm(out_nor_stage2_tf[:,:,:,0:3],axis=-1) * tf.norm(out_nor_stage2_tf[:,:,:,3:6],axis=-1)
 
  stage2_nor_flag_tf = tf.reduce_sum(out_nor_stage2_tf[:,:,:,0:3] * out_nor_stage2_tf[:,:,:,3:6],axis=-1,keep_dims=True)
  stage2_nor_flag_tf = tf.cast(stage2_nor_flag_tf < -0.5,tf.float32)
  stage2_nor_flag_tf = tf.reshape(stage2_nor_flag_tf,[FLAGS.batch_size,-1])

  top_k_stage2_feat_tf = top_k_feat_tf

  out_two_points_logits_tf, out_two_points_features_tf = two_point_quality(top_k_feat_tf, top_1024_feat_tf, FLAGS.batch_size, TOP_K, TOP_K2)

  out_two_points_score_tf = tf.nn.softmax(out_two_points_logits_tf)[:,:,1]

  gt_two_points_label_tf = tf.placeholder(tf.int32,[None, TOP_K * TOP_K2],'gt_label')
  gt_two_points_label_w_tf = tf.placeholder(tf.float32,[None, TOP_K * TOP_K2],'gt_label')

#### element wise loss
  mask_two_points_label_v2_tf = tf.cast(gt_two_points_label_tf,dtype=tf.float32) + tf.ones_like(gt_two_points_label_tf,dtype=tf.float32) * 0.01
  mask_two_points_label_tf = mask_two_points_label_v2_tf 

#### ListNet Loss
  out_two_points_score_tf_ = out_two_points_score_tf 
  denominator_stage2 = tf.reduce_sum(tf.exp(out_two_points_score_tf_),axis=1,keep_dims=True)
  loss_ListNet_Loss = - tf.reduce_mean(tf.cast(gt_two_points_label_tf,dtype=tf.float32) *  gt_two_points_label_w_tf * tf.log( tf.exp(out_two_points_score_tf_) / denominator_stage2 ) )
  loss_stage2 = loss_ListNet_Loss 
  
  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)

  out_two_points_score_tf = out_two_points_score_tf * stage2_nor_flag_tf
  out_two_points_top_value_tf, out_two_points_top_index_tf = tf.nn.top_k(out_two_points_score_tf, k=TOP_K2, sorted=True) 


with tf.variable_scope('stage3'):
  gripper_feat_stage3_tf = gripper_feat_tf#gripper_model_stage3(gripper_min_tf,gripper_max_tf)

  with tf.variable_scope('pointnet2'):
    nets_end = obj_model_v3(obj_pc_tf,is_training=True,num_class=2,bn_decay=None,gripper_feat=gripper_feat_stage2_tf,env_feat=None)
    out_single_point_feat_stage3_tf = nets_end['feats']

  out_single_point_top_1024_index_s3_tf = tf.placeholder(tf.int32,[None, TOP_K2])
  out_single_point_top_index_s3_tf = tf.placeholder(tf.int32,[None, TOP_K])
  out_two_points_top_index_s3_tf = tf.placeholder(tf.int32,[None, TOP_K2])

  if 1:
    out_single_point_top_1024_index_s3_tf =  out_single_point_top_1024_index_tf
    out_single_point_top_index_s3_tf = out_single_point_top_index_tf
    out_two_points_top_index_s3_tf = out_two_points_top_index_tf
    out_two_points_top_index_s3_tf = out_two_points_top_index_tf 

  top_k_nor_list = []
  top_1024_nor_list = []

  pred_pcn_tf_stage3 = tf.stop_gradient(pred_pcn_tf)

  for bi in range(FLAGS.batch_size):
    top_1024_nor_list.append(tf.gather(pred_pcn_tf_stage3[bi],out_single_point_top_1024_index_s3_tf[bi]))
    top_k_nor_list.append(tf.gather(pred_pcn_tf_stage3[bi],out_single_point_top_index_s3_tf[bi]))

  top_1024_nor_tf = tf.convert_to_tensor(top_1024_nor_list)
  top_k_nor_tf = tf.convert_to_tensor(top_k_nor_list)  


  top_k_feat_list_stage3 = []
  top_1024_feat_list_stage3 = []

  for bi in range(FLAGS.batch_size):
    top_1024_feat_list_stage3.append(tf.gather(out_single_point_feat_stage3_tf[bi], out_single_point_top_1024_index_s3_tf[bi]))
    top_k_feat_list_stage3.append(tf.gather(out_single_point_feat_stage3_tf[bi], out_single_point_top_index_s3_tf[bi]))
   
  top_1024_feat_tf_stage3 = tf.convert_to_tensor(top_1024_feat_list_stage3)
  top_k_feat_tf_stage3 = tf.convert_to_tensor(top_k_feat_list_stage3)

  top_1024_feat_tf_stage3 = tf.expand_dims(top_1024_feat_tf_stage3, axis=1)
  top_k_feat_tf_stage3 = tf.expand_dims(top_k_feat_tf_stage3,axis=2) 

  top_1024_feat_tf_stage3 = tf.tile(top_1024_feat_tf_stage3,[1, TOP_K, 1, 1])
  top_k_feat_tf_stage3 = tf.tile(top_k_feat_tf_stage3,[1, 1, TOP_K2, 1])
  
  out_two_points_features_stage3_tf = tf.concat([top_k_feat_tf_stage3, top_1024_feat_tf_stage3],axis=-1)
  out_two_points_features_stage3_tf = tf.reshape(out_two_points_features_stage3_tf,[FLAGS.batch_size, -1, 128])

  top_1024_xyz_tf = tf.expand_dims(top_1024_xyz_tf,axis=1)
  top_k_xyz_tf = tf.expand_dims(top_k_xyz_tf,axis=2)

  top_1024_xyz_tf = tf.tile(top_1024_xyz_tf,[1,TOP_K,1,1])
  top_k_xyz_tf = tf.tile(top_k_xyz_tf,[1,1,TOP_K2,1])

  out_xyz_stage2_tf = tf.concat([top_k_xyz_tf, top_1024_xyz_tf],axis=-1)
  out_xyz_stage2_tf = tf.reshape(out_xyz_stage2_tf,[FLAGS.batch_size, -1, 6])

  top_1024_nor_tf = tf.expand_dims(top_1024_nor_tf,axis=1)
  top_k_nor_tf = tf.expand_dims(top_k_nor_tf,axis=2)

  top_1024_nor_tf = tf.tile(top_1024_nor_tf,[1,TOP_K,1,1])
  top_k_nor_tf = tf.tile(top_k_nor_tf,[1,1,TOP_K2,1])

  out_nor_stage2_tf = tf.concat([top_k_nor_tf,top_1024_nor_tf],axis=-1)
  out_nor_stage2_tf = tf.reshape(out_nor_stage2_tf,[FLAGS.batch_size, -1, 6])


  top_k_feat_stage3_list = []
  top_1024_feat_stage3_list = []
  top_xyz_2point_list = []  
  top_nor_2point_list = []  


  for bi in range(FLAGS.batch_size):
    top_k_feat_stage3_list.append(tf.gather(out_two_points_features_stage3_tf[bi], out_two_points_top_index_s3_tf[bi][:TOP_K]))
    top_1024_feat_stage3_list.append(tf.gather(out_single_point_feat_stage3_tf[bi], out_single_point_top_1024_index_s3_tf[bi]))
    top_xyz_2point_list.append(tf.gather(out_xyz_stage2_tf[bi],out_two_points_top_index_tf[bi][:TOP_K])) 
    top_nor_2point_list.append(tf.gather(out_nor_stage2_tf[bi],out_two_points_top_index_tf[bi][:TOP_K])) 
 
  top_k_feat_stage3_tf = tf.convert_to_tensor(top_k_feat_stage3_list)
  top_1024_feat_stage3_tf = tf.convert_to_tensor(top_1024_feat_stage3_list)
  top_xyz_2point_tf = tf.convert_to_tensor(top_xyz_2point_list)
  top_nor_2point_tf = tf.convert_to_tensor(top_nor_2point_list)

  top_xyz_2point_tf = tf.expand_dims(top_xyz_2point_tf,axis=2)
  top_xyz_2point_tf = tf.tile(top_xyz_2point_tf,[1,1,TOP_K2,1])

  top_xyz_3point_tf = tf.concat([top_xyz_2point_tf, top_1024_xyz_tf],axis=-1)

  top_nor_2point_tf = tf.expand_dims(top_nor_2point_tf,axis=2)
  top_nor_2point_tf = tf.tile(top_nor_2point_tf,[1,1,TOP_K2,1])
  top_nor_3point_tf = tf.concat([top_nor_2point_tf, top_1024_nor_tf],axis=-1)

  print("*****************:")
  print(top_nor_3point_tf) 

  top_nor_n12_tf = tf.reduce_sum(top_nor_3point_tf[:,:,:,0:3]*top_nor_3point_tf[:,:,:,3:6],axis=-1,keep_dims=True)
  top_nor_n23_tf = tf.reduce_sum(top_nor_3point_tf[:,:,:,3:6]*top_nor_3point_tf[:,:,:,6:9],axis=-1,keep_dims=True)
  top_nor_n31_tf = tf.reduce_sum(top_nor_3point_tf[:,:,:,0:3]*top_nor_3point_tf[:,:,:,6:9],axis=-1,keep_dims=True)

  print("top_nor_n12_tf",top_nor_n12_tf)
  
  thre = 0.6
  type1 = tf.cast(tf.reduce_sum(tf.cast(tf.concat([top_nor_n12_tf < -thre , top_nor_n31_tf < -thre , top_nor_n23_tf > thre],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 2,tf.float32)
  type2 = tf.cast(tf.reduce_sum(tf.cast(tf.concat([top_nor_n12_tf < -thre, top_nor_n31_tf > thre , top_nor_n23_tf < -thre],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 2,tf.float32)
  type3 = tf.cast(tf.reduce_sum(tf.cast(tf.concat([top_nor_n12_tf > thre , top_nor_n31_tf < -thre , top_nor_n23_tf < -thre],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 2,tf.float32)

  type_flag_tf = tf.cast(tf.reduce_sum(tf.cast(tf.concat([type1,type2,type3],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 0,tf.float32)
  print("type1",type1)
  type_flag_tf = tf.reshape(type_flag_tf,(FLAGS.batch_size,-1)) 

  f1t2 = (top_xyz_3point_tf[:,:,:,3:6] - top_xyz_3point_tf[:,:,:,0:3]) 
  f2t3 = (top_xyz_3point_tf[:,:,:,6:9] - top_xyz_3point_tf[:,:,:,3:6])
  f3t1 = (top_xyz_3point_tf[:,:,:,0:3] - top_xyz_3point_tf[:,:,:,6:9])

  f2t1 = -f1t2
  f3t2 = -f2t3
  f1t3 = -f3t1

  fc_thre = math.sin(math.atan(0.42))

  nf2t1 = f2t1 / (tf.norm(f2t1,axis=-1,keep_dims=True)+1e-5)
  nf3t1 = f3t1 / (tf.norm(f3t1,axis=-1,keep_dims=True)+1e-5)
  nf2t3 = f2t3 / (tf.norm(f2t3,axis=-1,keep_dims=True)+1e-5)

  nf1t2 = -nf2t1
  nf1t3 = -nf3t1
  nf3t2 = -nf2t3
########################
#################################
###########################
  v1_23 = tf.cross(top_nor_3point_tf[:,:,:,0:3],f2t3)
  v1_23 = v1_23 / (tf.norm(v1_23,axis=-1,keep_dims=True)+1e-5)

  v2_13 = tf.cross(top_nor_3point_tf[:,:,:,3:6],f1t3)
  v2_13 = v2_13 / (tf.norm(v2_13,axis=-1,keep_dims=True)+1e-5)

  v3_12 = tf.cross(top_nor_3point_tf[:,:,:,6:9],f1t2)
  v3_12 = v3_12 / (tf.norm(v3_12,axis=-1,keep_dims=True)+1e-5)


  fc_check1 = type1 * tf.cast(tf.abs(tf.reduce_sum(nf2t1 * v1_23,axis=-1,keep_dims=True)) < fc_thre,tf.float32) *  tf.cast(tf.abs(tf.reduce_sum(f3t1 * v1_23,axis=-1,keep_dims=True)) < fc_thre, tf.float32)
  check_check = tf.reduce_sum(nf2t1 * top_nor_3point_tf[:,:,:,0:3],axis=-1,keep_dims=True) > 0.5
  fc_check2 = type2 * tf.cast(tf.abs(tf.reduce_sum(nf1t2 * v2_13,axis=-1,keep_dims=True)) < fc_thre,tf.float32) *  tf.cast(tf.abs(tf.reduce_sum(f3t2 * v2_13,axis=-1,keep_dims=True)) < fc_thre, tf.float32)
  fc_check3 = type3 * tf.cast(tf.abs(tf.reduce_sum(nf1t3 * v3_12,axis=-1,keep_dims=True)) < fc_thre,tf.float32) *  tf.cast(tf.abs(tf.reduce_sum(f2t3 * v3_12,axis=-1,keep_dims=True)) < fc_thre, tf.float32)

  fc_check_tf = tf.cast(tf.reduce_sum(tf.cast(tf.concat([fc_check1,fc_check2,fc_check3],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 0,tf.float32)
  fc_check_tf = tf.reshape(fc_check_tf,(FLAGS.batch_size,-1))



  check1 = tf.minimum(top_nor_3point_tf[:,:,:,0:3] * f1t2,top_nor_3point_tf[:,:,:,0:3] * f1t3) < 0
  check2 = tf.minimum(top_nor_3point_tf[:,:,:,3:6] * f2t1,top_nor_3point_tf[:,:,:,0:3] * f2t3) < 0
  check3 = tf.minimum(top_nor_3point_tf[:,:,:,6:9] * f3t2,top_nor_3point_tf[:,:,:,0:3] * f3t1) < 0
  check_flag_tf = tf.cast(tf.reduce_sum(tf.cast(tf.concat([check1,check2,check3],axis=-1),tf.uint8),axis=-1,keep_dims=True) > 2,tf.float32) 
  check_flag_tf = tf.reshape(check_flag_tf,(FLAGS.batch_size,-1))

  top_triangle_side1 = tf.linalg.norm((top_xyz_3point_tf[:,:,:,0:3] - top_xyz_3point_tf[:,:,:,3:6]),axis=3,keepdims=True)
  top_triangle_side2 = tf.linalg.norm((top_xyz_3point_tf[:,:,:,0:3] - top_xyz_3point_tf[:,:,:,6:9]),axis=3,keepdims=True)
  top_triangle_side3 = tf.linalg.norm((top_xyz_3point_tf[:,:,:,6:9] - top_xyz_3point_tf[:,:,:,3:6]),axis=3,keepdims=True)

  top_triangle_side = tf.concat([top_triangle_side1,top_triangle_side2,top_triangle_side3],axis=-1)
  
  min_side_tf = tf.reduce_min(top_triangle_side,axis=-1)
  flag_side = tf.cast(min_side_tf > 0.01,tf.float32)
  flag_side_tf = tf.reshape(flag_side,(FLAGS.batch_size,-1))

  max_side_tf = tf.reduce_max(top_triangle_side,axis=-1)
  max_flag_side_tf = tf.reshape(max_side_tf,(FLAGS.batch_size,-1))
  weight_max_side = tf.exp(-max_flag_side_tf * 10.0) + 0.2
 
  top_triangle_a1 =  (top_triangle_side2 * top_triangle_side2 + top_triangle_side3  * top_triangle_side3 - top_triangle_side1 * top_triangle_side1) / (top_triangle_side2* top_triangle_side3 * 2 + 1e-10)
  top_triangle_a2 =  (top_triangle_side1 * top_triangle_side1 + top_triangle_side3  * top_triangle_side3 - top_triangle_side2 * top_triangle_side2) / (top_triangle_side1* top_triangle_side3 * 2 + 1e-10)
  top_triangle_a3 =  (top_triangle_side1 * top_triangle_side1 + top_triangle_side2  * top_triangle_side2 - top_triangle_side3 * top_triangle_side3) / (top_triangle_side2* top_triangle_side1 * 2 + 1e-10)

  flag_triangle = tf.concat([top_triangle_a1, top_triangle_a2, top_triangle_a3],axis=-1)
  flag_triangle = flag_triangle - tf.cast(flag_triangle > 1.0,dtype=tf.float32) * (flag_triangle - 0.99) - tf.cast(flag_triangle < -1.0,tf.float32) * (-1.01 - flag_triangle)
  angle_degree_tf = tf.acos(flag_triangle)

  max_angle_tf = tf.reduce_min(flag_triangle,axis=-1)
  max_angle_flag = tf.cast(max_angle_tf > -0.5,tf.float32)
  max_angle_degree_tf = tf.math.acos(max_angle_tf)
  max_angle_degree_tf = tf.reshape(max_angle_degree_tf,(FLAGS.batch_size,-1))
  max_angle_degree_tf = tf.stop_gradient(max_angle_degree_tf)

  angle_flag_tf = tf.reshape(max_angle_flag,(FLAGS.batch_size,-1))

  out_corr_logits_stage3_tf, out_corr_features_stage3_tf = two_point_quality_stage3(top_k_feat_stage3_tf, top_1024_feat_stage3_tf, FLAGS.batch_size, TOP_K, TOP_K2)#, top_nor_3point_tf)

  out_corr_score_stage3_tf = tf.nn.softmax(out_corr_logits_stage3_tf)[:,:,1]
  gt_corr_label_stage3_tf = tf.placeholder(tf.int32,[None, TOP_K * TOP_K2],'gt_label')

  print("out_corr_score_stage3_tf",out_corr_score_stage3_tf)
  final_mask_tf = type_flag_tf * angle_flag_tf * flag_side_tf * check_flag_tf * fc_check_tf
  final_mask_gt_tf = tf.cast(gt_corr_label_stage3_tf,tf.float32) * final_mask_tf

  gt_corr_label_stage3_tf_mask = tf.cast(gt_corr_label_stage3_tf,dtype=tf.float32) 

####ListNet
  out_corr_score_stage3_tf_ = out_corr_score_stage3_tf
  denominator_stage3 = tf.reduce_sum(tf.exp(out_corr_score_stage3_tf_),axis=1,keep_dims=True)
  loss_ListNet_Loss_s3 = - tf.reduce_mean( gt_corr_label_stage3_tf_mask * tf.log( tf.exp(out_corr_score_stage3_tf_) / denominator_stage3 ) )
 
  loss_stage3 = loss_ListNet_Loss_s3 * 1.0

  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)

  out_corr_score_stage3_tf = out_corr_score_stage3_tf * final_mask_tf

  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 


init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init_op)


SAVER = tf.train.Saver(max_to_keep=1000)

def copy_stage3_obj():
  in_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stage2/pointnet2')
  out_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stage3/pointnet2')
  stage2_op = [l_p.assign(g_p) for  l_p, g_p in zip(out_params, in_params)]
  sess.run(stage2_op)


def copy_stage2_obj():
  in_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stage1/pointnet2')
  out_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stage2/pointnet2')
  stage2_op = [l_p.assign(g_p) for  l_p, g_p in zip(out_params, in_params)]
  sess.run(stage2_op)


def restore_stage2_v2(epoch):
  save_top_dir = os.path.join('../saved_models',"point_set_selection")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  variables = slim.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0]=='stage1']
  saver = tf.train.Saver(variables_to_restore)
  print("restoring from %s" % ckpt_path)
  saver.restore(sess, ckpt_path)
 

def restore_stage3_v1(epoch):
  save_top_dir = os.path.join('../saved_models',"point_set_selection")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  variables = slim.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0]=='stage1']
  saver = tf.train.Saver(variables_to_restore)
  print("restoring from %s" % ckpt_path)
  saver.restore(sess, ckpt_path)

def restore_stage3_v2(epoch):
  save_top_dir = os.path.join('../saved_models',"point_set_selection")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  variables = slim.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0]=='stage1' or v.name.split('/')[0] == 'stage2']
  saver = tf.train.Saver(variables_to_restore)
  print("restoring from %s" % ckpt_path)
  saver.restore(sess, ckpt_path)
  #SAVER.restore(sess, ckpt_path)
  #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  #print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name="", all_tensors=False, all_tensor_names=True)


def restore_stage3(epoch):
  save_top_dir = os.path.join('../saved_models',"point_set_selection")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)
  #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  #print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name="", all_tensors=False, all_tensor_names=True)


def save_model_stage3(epoch):
  save_top_dir = os.path.join('../saved_models',"point_set_selection")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  if epoch == 0:
    SAVER.save(sess, ckpt_path, write_meta_graph=True)
  else:
    SAVER.save(sess, ckpt_path, write_meta_graph=False)
  print("Saving model at epoch %d to %s" % (epoch, ckpt_path))


def test(base,pc_real):
  nnn = 1
  gripper_size = nnn
  num_batch = 1
  for batch_id in range(int(num_batch)):
      in_gripper_feat_list = []
      in_objenv_list = []
      in_objnor_list = []
      gripper_id_list = []

      gripper_max_list = []
      gripper_mean_list = []
      gripper_min_list = []

      old_id_new_list = []

      gripper_index = np.array([11])
      #in_gripper_id = 2
      #input_gripper_index = np.array([5])
      #gripper_index = np.array([11])#np.random.choice(np.array([1,2,3,4,5,7,8,9,11,12,13]),FLAGS.batch_size,replace=True) 
      rra =  np.random.uniform(0,1)

      for bbi in range(int(1)):
        rotation_degree_array = np.arange(0,360,30)
        rotation_degree = np.random.choice(rotation_degree_array,1)
        rotmat = np.zeros((3,3))
        rotmat[0,0] = np.cos(rotation_degree)
        rotmat[0,1] = -np.sin(rotation_degree)
        rotmat[1,0] = np.sin(rotation_degree)
        rotmat[1,1] = np.cos(rotation_degree)
        rotmat[2,2] = 1
        translx, transly, translz = np.random.uniform(-0.05,0.05,size=(3,))
        transl = np.array([translx, transly, translz / 5.0])
          
        objenv_list = []
        for ggi in range(gripper_size):
          gripper_id = gripper_index[ggi]
          bi = bbi * gripper_size + ggi
          gripper_id_list.append(gripper_id)
          gripper_id = str(gripper_id)

          if int(gripper_id) < 11:
            #in_gripper_name = 'robotiq_2f'
            #gripper_path_mean = os.path.join(GRIPPER_TOP_DIR,in_gripper_name,'mean.npy')
            #gripper_path_max = os.path.join(GRIPPER_TOP_DIR,in_gripper_name,'max.npy')
            #gripper_path_min = os.path.join(GRIPPER_TOP_DIR,in_gripper_name,'min.npy')

            #gripper_path_mean = os.path.join(GRIPPER_TOP_DIR,'G'+str(in_gripper_id),'mean.npy')
            #gripper_path_max = os.path.join(GRIPPER_TOP_DIR,'G'+str(in_gripper_id),'max.npy')
            #gripper_path_min = os.path.join(GRIPPER_TOP_DIR,'G'+str(in_gripper_id),'min.npy')

            gripper_path_mean = os.path.join(GRIPPER_TOP_DIR,'G'+str(gripper_id),'mean.npy')
            gripper_path_max = os.path.join(GRIPPER_TOP_DIR,'G'+str(gripper_id),'max.npy')
            gripper_path_min = os.path.join(GRIPPER_TOP_DIR,'G'+str(gripper_id),'min.npy')

            gripper_feat_mean = np.load(gripper_path_mean)
            gripper_feat_max = np.load(gripper_path_max)
            gripper_feat_min = np.load(gripper_path_min)
            gripper_feat = np.hstack([gripper_feat_mean,gripper_feat_max,gripper_feat_min])[0]
            in_gripper_feat_list.append(gripper_feat)
          else:
            if 1:
                gripper_name = 'None'
                if int(gripper_id) == 12:
                  gripper_ends_with = '_par_bh282tmp_label_stage1.npy'
                  gripper_name = 'bh_282'
                elif int(gripper_id) == 11:
                  gripper_ends_with = '_par_robotiq_3f_fullest_tmp_label_stage1.npy'
                  gripper_name = 'robotiq_3f'
                elif int(gripper_id) == 13:
                  gripper_ends_with = '_par_kinova_3f_fullest_tmp_labelkinova_stage1.npy'
                  gripper_name = 'kinova_kg3'
                gripper_path_mean = os.path.join(GRIPPER_TOP_DIR,str(gripper_name),'mean.npy')
                gripper_path_max = os.path.join(GRIPPER_TOP_DIR,str(gripper_name),'max.npy')
                gripper_path_min = os.path.join(GRIPPER_TOP_DIR,str(gripper_name),'min.npy')

                gripper_feat_mean = np.load(gripper_path_mean)
                gripper_feat_max = np.load(gripper_path_max)
                gripper_feat_min = np.load(gripper_path_min)
                gripper_feat = np.hstack([gripper_feat_mean,gripper_feat_max,gripper_feat_min])[0]
                in_gripper_feat_list.append(gripper_feat)

          in_objenv_list.append(pc_real)

      in_gripper_feat = np.array(in_gripper_feat_list)
      in_objenv = np.array(in_objenv_list)
      print("in_objenv",in_objenv.shape)

      for _ in range(30):
        pred_label, out_single_point_top_1024_index ,out_single_point_top_index = sess.run([pred_label_tf, out_single_point_top_1024_index_tf, out_single_point_top_index_tf],feed_dict={gripper_feat_tf:in_gripper_feat, obj_pc_tf: in_objenv})

        if 0:
          for gj in range(1):
            s_p = np.copy(in_objenv[gj])
            s_p[:,2] *= -1.0
            c_c = np.zeros((2048,3))
            c_c[:,0] = 255.0#194.0
            pred_c = out_single_point_top_index[gj][0:120]
            c_c[pred_c,0] = 0.0# Prediction Red
            c_c[pred_c,1] = 255.0
            showpoints(s_p,c_gt=c_c,waittime=5,freezerot=False) ### GRB
            #input("raw")

        # stage2
        if 1:
          out_single_point_top_1024_index_v2, out_two_points_top_index = sess.run([out_single_point_top_1024_index_tf, out_two_points_top_index_tf],feed_dict={gripper_feat_tf: in_gripper_feat, obj_pc_tf: in_objenv})

          assert np.all(out_single_point_top_1024_index_v2 == out_single_point_top_1024_index)    

          for gj in range(1):
            two_points_label =  out_two_points_top_index[gj]

            top_two_points_index_1 = two_points_label // TOP_K2
            top_two_points_index_1 = out_single_point_top_index[gj][top_two_points_index_1]
            top_two_points_index_2 = out_single_point_top_1024_index[gj][two_points_label % TOP_K2]
   
            top1_two_points_index_1 = top_two_points_index_1[0:12]
            top1_two_points_index_2 = top_two_points_index_2[0:12]

            gtt = np.vstack([[top1_two_points_index_1],[top1_two_points_index_2]]).T[:50]
            
            if 0:
              s_p = np.copy(in_objenv[gj])
              s_p[:,2] *= -1.0
              c_pred = np.zeros((2048,3))
              c_pred[:,0] = 255.0#255#194.0
              c_pred[top1_two_points_index_1,0] = 0.0
              c_pred[top1_two_points_index_1,1] = 255.0
              c_pred[top1_two_points_index_1,2] = 0.0
              c_pred[top1_two_points_index_2,0] = 0.0
              c_pred[top1_two_points_index_2,2] = 255.0
              showpoints(s_p,c_gt=c_pred,waittime=20,ballradius=4,freezerot=False) ## GRB
              #input("raw") 
     
        # stage3 oldr
        if 1:
          out_corr_top_index_stage3 = sess.run(out_corr_top_index_stage3_tf,feed_dict={gripper_feat_tf: in_gripper_feat, obj_pc_tf: in_objenv})
        
          for gj in range(nnn):
            third_point_set_label = out_corr_top_index_stage3[gj]

            top_corr2_index = third_point_set_label // TOP_K2
       
            two_points_label_ = out_two_points_top_index[gj] 
            top_two_points_index_1_ = two_points_label_ // TOP_K2
            top_two_points_index_1_ = out_single_point_top_index[gj][top_two_points_index_1_]
            top_two_points_index_2_ = out_single_point_top_1024_index[gj][two_points_label_ % TOP_K2]
          
            top_f1_index_1 = top_two_points_index_1_[top_corr2_index]
            top_f2_index_2 = top_two_points_index_2_[top_corr2_index]
            top_f3_index_3 = out_single_point_top_1024_index[gj][third_point_set_label % TOP_K2]
     
            top1_f1_index_1 = top_f1_index_1[0:12]
            top1_f2_index_2 = top_f2_index_2[0:12]
            top1_f3_index_3 = top_f3_index_3[0:12]

            gggt = np.vstack([[top1_f1_index_1],[top1_f2_index_2],[top1_f3_index_3]]).T

            #if 1:
            s_p = np.copy(in_objenv[0])
            s_p[:,2] *= -1.0
            c_pred = np.zeros((2048,3))
            c_pred[:,0] = 255.0#255#194.0
            c_pred[:,1] = 255#194.0
            c_pred[:,2] = 255.0#214.0
            c_pred[top1_f1_index_1,0] = 0.0
            c_pred[top1_f1_index_1,1] = 255.0
            c_pred[top1_f1_index_1,2] = 0.0
            c_pred[top1_f2_index_2,0] = 255.0
            c_pred[top1_f2_index_2,1] = 0.0
            c_pred[top1_f2_index_2,2] = 0.0
            c_pred[top1_f3_index_3,0] = 0.0
            c_pred[top1_f3_index_3,1] = 0.0
            c_pred[top1_f3_index_3,2] = 255.0
            showpoints(s_p,c_gt=c_pred,waittime=5,ballradius=4,freezerot=False) ## GRB
            #input("r")
 

if __name__ == "__main__":
  restore_stage3(220)
  pc_dir = "../data/real_world/d61.npy"
  pc_real = np.load(pc_dir)
  test(0,pc_real)
