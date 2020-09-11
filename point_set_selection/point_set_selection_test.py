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

from data_preparing_test import train_val_test_list
print("train num %d , val num %d , test num %d" % (len(train_val_test_list._train),len(train_val_test_list._val),len(train_val_test_list._test)))

nnn = 2
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


DATA_TOP_DIR = os.path.join('/juno/u/lins2/MetaGrasp/','Data','BlensorResult_test')

GRIPPER_TOP_DIR = os.path.join(ROOT_DIR,'Data','Gripper_DB')

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
  save_top_dir = os.path.join('./saved_models',"point_set_selection_final")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  variables = slim.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0]=='stage1']
  saver = tf.train.Saver(variables_to_restore)
  print("restoring from %s" % ckpt_path)
  saver.restore(sess, ckpt_path)
 

def restore_stage3_v1(epoch):
  save_top_dir = os.path.join('./saved_models',"point_set_selection_final")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  variables = slim.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0]=='stage1']
  saver = tf.train.Saver(variables_to_restore)
  print("restoring from %s" % ckpt_path)
  saver.restore(sess, ckpt_path)

def restore_stage3_v2(epoch):
  save_top_dir = os.path.join('./saved_models',"point_set_selection_final")
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
  save_top_dir = os.path.join('./saved_models',"point_set_selection_final")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)
  #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  #print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name="", all_tensors=False, all_tensor_names=True)


def save_model_stage3(epoch):
  save_top_dir = os.path.join('./saved_models',"point_set_selection_single")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  if epoch == 0:
    SAVER.save(sess, ckpt_path, write_meta_graph=True)
  else:
    SAVER.save(sess, ckpt_path, write_meta_graph=False)
  print("Saving model at epoch %d to %s" % (epoch, ckpt_path))


def test(base=0):
  num_examples = len(train_val_test_list._train)
  gripper_size = nnn
  print("num training examples %d" % (num_examples))
  num_batch = int(num_examples/int(FLAGS.batch_size/gripper_size))
  print("num_training_batch", num_batch)

  for epoch in range(1):
    train_loss = 0.0
    train_loss_stage2 = 0.0
    train_loss_stage3 = 0.0
    train_loss_nor = 0.0
    num_batch_stage3 = 0
    score_stage2 = 0

    stage1_acc_top1 = 0
    stage1_acc_top10 = 0

    stage2_acc_top1 = 0
    stage2_acc_top10 = 0
    gt_stage2_acc_top1 = 0
 
    stage3_acc_top1 = 0
    stage3_acc_top10 = 0

    gt_stage3_acc_top1 = 0
    gt_stage3_acc_top30 = 0

    final_neighbor = 0

    train_index = np.random.permutation(num_examples) 
    for batch_id in range(int(num_batch)):
      in_gripper_feat_list = []
      in_objenv_list = []
      in_objnor_list = []
      gt_gq_list = [] 
      gt_label_list = []
      bt_index = train_index[batch_id * int(FLAGS.batch_size / gripper_size): (batch_id + 1) * int(FLAGS.batch_size / gripper_size)] 
      gripper_id_list = []
      gt_env_num_list = []
      gt_env_id_list = []
      gt_s1_weight_list = []
      gt_gt_list = []

      gripper_max_list = []
      gripper_mean_list = []
      gripper_min_list = []

      old_id_new_list = []

      #gripper_index = np.random.choice(np.array([1,2,3,4,5,11,12,13]),FLAGS.batch_size,replace=True) 
      gripper_index = np.array([3,5])
      #in_gripper_id = 2
      #input_gripper_index = np.array([5])
      #gripper_index = np.array([11])#np.random.choice(np.array([1,2,3,4,5,7,8,9,11,12,13]),FLAGS.batch_size,replace=True) 
      rra =  np.random.uniform(0,1)

      for bbi in range(int(FLAGS.batch_size/gripper_size)):
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

          
        stage1_gq_label_list = []
        objenv_list = []
        objnor_list = []
        old_id_new_gripper_list = []
        stage1_gq_label_weight_list  = []
 
        for ggi in range(gripper_size):
          gripper_id = gripper_index[ggi]
          bi = bbi * gripper_size + ggi
          gripper_id_list.append(gripper_id)
          gripper_id = str(gripper_id)

          env_i = str(train_val_test_list._train[bt_index[bbi]])
          env_dir = os.path.join(DATA_TOP_DIR, env_i)
          obj_path = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith('_pcn_new_normal.npz.npy')][0]
          obj_pcs = np.load(obj_path)
          
          gt_env_path = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith('_pc_above_table.npy')][0]
          gt_env_id = np.load(gt_env_path)
          gt_env_id_list.append(gt_env_id)
          gt_env_num = len(gt_env_id)
          gt_env_num_list.append(gt_env_num)

          old_id_new = np.arange(2048)
          old_id_new[0:gt_env_num] = gt_env_id
          stage1_gq_label_weight = np.ones((2048,))

          if int(gripper_id) < 11:
            gt_path_endswith = '_par_grasp' + str(gripper_id) + '.npz'#'_new.npz'
            gt_gq_path_dirs = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith(gt_path_endswith)]
            gt_gq_path = gt_gq_path_dirs[0]
            gt_gq_label = np.load(gt_gq_path)['par']
            gt_label = np.zeros((2048,))
            stage1_gq_label = np.zeros((2048,))
            if len(gt_gq_label) > 0:
              gt_label_1 = np.unique(gt_gq_label[:,5,1])
              gt_label_2 = np.unique(gt_gq_label[:,5,2])
              gt_label_12 = np.unique(np.hstack([gt_label_1,gt_label_2])).astype(np.int32)
              gt_label_filter = [l for l in gt_label_12 if l in gt_env_id]
              gt_label_filter = np.array(gt_label_filter)
              gt_label[gt_label_filter] = 1
              stage1_gq_label[0:gt_env_num] = gt_label[gt_env_id]

            GRIPPER_TOP_DIR = '/juno/u/lins2/MetaGrasp/Data/Gripper/Data_DB'
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
            gt_label = np.zeros((2048,))#np.ones((2048,))#np.zeros((2048,))
            gt_label_weight = np.zeros((2048,))
            stage1_gq_label = np.zeros((2048,))
            stage1_gq_label_weight = np.zeros((2048,))
            if 1:
                gt_label_123_dir = os.path.join(DATA_TOP_DIR,env_i)
                GRIPPER_TOP_DIR = '/juno/u/lins2/MetaGrasp/Data/Gripper/Data_DB'
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
                assert os.path.exists(gripper_path_mean)
                assert os.path.exists(gripper_path_max)
                assert os.path.exists(gripper_path_min)

                gripper_feat_mean = np.load(gripper_path_mean)
                gripper_feat_max = np.load(gripper_path_max)
                gripper_feat_min = np.load(gripper_path_min)
                gripper_feat = np.hstack([gripper_feat_mean,gripper_feat_max,gripper_feat_min])[0]
                in_gripper_feat_list.append(gripper_feat)
                
                gt_label_123_dirs = [line for line in os.listdir(gt_label_123_dir) if line.endswith(gripper_ends_with)]
                if len(gt_label_123_dirs) == 0:
                  print(gripper_ends_with,gt_label_123_dir)

                gt_label_123_dir = os.path.join(gt_label_123_dir,gt_label_123_dirs[0])
                gt_label_123 = np.load(gt_label_123_dir)
                gt_label_123_weight_dir = gt_label_123_dir.split('stage1.npy')[0]+'stage1_w.npy'

                gt_label_123_weight  = np.load(gt_label_123_weight_dir) 
                if len(gt_label_123) > 0:
                  gt_label[0:gt_env_num] = gt_label_123[gt_env_id_list[bi]]
                  gt_label_weight[0:gt_env_num] = gt_label_123_weight[gt_env_id_list[bi]]
                  gt_label_weight = gt_label_weight / (np.max(gt_label_weight) * 0.5 +1e-5) + 0.5
                stage1_gq_label[0:gt_env_num] = gt_label[:gt_env_num]
                stage1_gq_label_weight[0:gt_env_num] = gt_label_weight[0:gt_env_num] 
    
          objenv = np.zeros((2048,3))
          objenv[0:gt_env_num] = obj_pcs[0:gt_env_num,:3]
          
          objnor = np.zeros((2048,3))
          objnor[0:gt_env_num] = obj_pcs[0:gt_env_num,3:]

          stage1_gq_label_list.append(stage1_gq_label)
          old_id_new_gripper_list.append(old_id_new)
          objnor_list.append(objnor)
          objenv_list.append(objenv)
          stage1_gq_label_weight_list.append(stage1_gq_label_weight)

        stage1_gq_label_total = np.zeros((2048,))

        for ggi in range(gripper_size):
          stage1_gq_label_total = np.logical_or(stage1_gq_label_list[ggi], stage1_gq_label_total)

        selected = np.where(stage1_gq_label_total[0:gt_env_num] < 1)[0]
        if len(selected) > 0:
          
          selected_id = np.random.choice(selected,2048-gt_env_num,replace=True)
          for ggi in range(gripper_size):
            tmp_objenv = np.copy(objenv_list[ggi])
            tmp_objenv[gt_env_num:] = tmp_objenv[selected_id]

            aug_flag = np.random.uniform(0,1)
            if aug_flag < -1.5:
              tmp_objenv = tmp_objenv.dot(rotmat.transpose()) + transl
            in_objenv_list.append(tmp_objenv)

            tmp_objnor = np.copy(objnor_list[ggi])
            tmp_objnor[gt_env_num:] = tmp_objnor[selected_id]
            if aug_flag < -1.5:
              tmp_objnor =  tmp_objnor.dot(rotmat.transpose())
            in_objnor_list.append(tmp_objnor)

            tmp_stage1_gq_label = np.copy(stage1_gq_label_list[ggi])
            tmp_stage1_gq_label[gt_env_num:] = tmp_stage1_gq_label[selected_id]
            gt_gq_list.append(tmp_stage1_gq_label)

            tmp_old_id_new = np.copy(old_id_new_gripper_list[ggi])
            tmp_old_id_new[gt_env_num:] = tmp_old_id_new[selected_id]
            old_id_new_list.append(tmp_old_id_new)

            gt_s1_weight_list.append(stage1_gq_label_weight_list[ggi])
        else:
          print(env_dir) 
       
      in_gripper_feat = np.array(in_gripper_feat_list)
      in_objenv = np.array(in_objenv_list)
      in_objnor = np.array(in_objnor_list)
      stage1_gt_gq = np.array(gt_gq_list)
      stage1_gt_gq_weight = np.array(gt_s1_weight_list)

      lossnor, loss_value, pred_label, out_single_point_top_1024_index ,out_single_point_top_index = sess.run([loss_nor, loss_stage1, pred_label_tf, out_single_point_top_1024_index_tf, out_single_point_top_index_tf],feed_dict={gq_label_weight_tf: stage1_gt_gq_weight, gt_pcn_tf:in_objnor,  gripper_feat_tf:in_gripper_feat, obj_pc_tf: in_objenv, gq_label_tf:stage1_gt_gq})
      train_loss_nor += lossnor

      for gj in range(FLAGS.batch_size): 
        stage1_top1 = out_single_point_top_index[gj][0:1] 
        stage1_top10 = out_single_point_top_index[gj][0:10] 

        if np.sum(stage1_gt_gq[gj][stage1_top1]) > 0:
          stage1_acc_top1 += 1.0
        else:
          p1 = in_objenv[gj][stage1_top1]
          stree = KDTree(in_objenv[gj])
          ind1 = stree.query_radius(p1,r=0.005)[0]
          iflag = 0
          for nd1 in ind1:
            if stage1_gt_gq[gj][nd1]:
              iflag = 1
              break
          stage1_acc_top1 += iflag
        if np.sum(stage1_gt_gq[gj][stage1_top10]) > 0:
          stage1_acc_top10 += 1.0
        else:
          for ppp in range(10):
            p1 = in_objenv[gj][stage1_top10[ppp:ppp+1]]
            stree = KDTree(in_objenv[gj])
            ind1 = stree.query_radius(p1,r=0.005)[0]
            iflag = 0
            for nd1 in ind1:
              if stage1_gt_gq[gj][nd1]:
                iflag = 1
                break
            if iflag:
              stage1_acc_top10 += 1.0
              break

        #stage1_acc_top1 += stage1_gt_gq[gj][stage1_top1]
        #stage1_acc_top10 += (np.sum(stage1_gt_gq[gj][stage1_top10]) > 0) 
 
      train_loss += loss_value
      if 1:
       for gj in range(FLAGS.batch_size):
        s_p = np.copy(in_objenv[gj])
        s_p[:,2] *= -1.0
        gt_c = np.where(stage1_gt_gq[gj] > 0.5)[0]
        c_c = np.zeros((2048,3))
        c_c[:,0] = 255.0#194.0
        c_c[gt_c,0] = 0.0#255.0 # Ground Truth Dark Blue
        c_c[gt_c,1] = 0.0#102.0
        c_c[gt_c,2] = 255.0#51.0
        pred_c = out_single_point_top_index[gj][0:120]
        c_c[pred_c,0] = 0.0# Prediction Red
        c_c[pred_c,1] = 255.0
        showpoints(s_p,c_gt=c_c,waittime=5,freezerot=False) ### GRB
        input("raw")

      # stage2
      if 1:
        gt_two_points_label_list = []
        gt_two_points_label_w_list = []
        for bbi in range(int(FLAGS.batch_size/gripper_size)):
          for ggi in range(gripper_size):
            gripper_id = gripper_index[ggi]
            bi = bbi * gripper_size + ggi
            gripper_path = GRIPPER_TOP_DIR
            env_i =  str(train_val_test_list._train[bt_index[bbi]])
            env_dir = os.path.join(DATA_TOP_DIR, env_i)
            tmp_label = np.zeros((2048,2048))      
            if gripper_id < 11:
              gt_path_endswith = '_par_grasp'+ str(gripper_id) + '.npz'#'_new.npz'
              gt_gq_path_dirs = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith(gt_path_endswith)]
              gt_gq_path = gt_gq_path_dirs[0]
              gt_gq_label = np.load(gt_gq_path)['par']
              gt_label = np.zeros((TOP_K, 1024))
              if len(gt_gq_label) > 0:
                out_single_index_stage2 = [(idx, gt_env_id_list[bi][l]) for idx, l in enumerate(out_single_point_top_1024_index[bi]) if l < gt_env_num_list[bi]]
                out_single_index_stage2_tmp = np.array(out_single_index_stage2)
                tmp_label = np.zeros((2048,2048))
                tmp_label[np.copy(gt_gq_label[:,5,1]).astype(np.int32),np.copy(gt_gq_label[:,5,2]).astype(np.int32)] = 1
                tmp_label[np.copy(gt_gq_label[:,5,2]).astype(np.int32),np.copy(gt_gq_label[:,5,1]).astype(np.int32)] = 1
                for k in range(TOP_K):
                  if out_single_point_top_index[bi][k] < gt_env_num_list[bi]:# and stage1_gt_gq[bi][out_single_point_top_index[bi][k]] > 0:
                    gt_label[k][out_single_index_stage2_tmp[:,0]] = tmp_label[gt_env_id_list[bi][out_single_point_top_index[bi][k]],out_single_index_stage2_tmp[:,1]]
              gt_two_points_label_list.append(gt_label)
              gt_label_w = np.ones((TOP_K,TOP_K2))
              gt_two_points_label_w_list.append(gt_label_w)
            else:
              tmp_label = np.zeros((2048,2048))
              gt_label = np.zeros((TOP_K, TOP_K2))
              gt_label_w = np.zeros((TOP_K,TOP_K2))

              if 1:#len(gt_3f_label) > 0:
                  out_single_index_stage2 = [(idx, old_id_new_list[bi][l]) for idx, l in enumerate(out_single_point_top_1024_index[bi])]
                  out_single_index_stage2_tmp = np.array(out_single_index_stage2)

                  tmp_label_dir = os.path.join(DATA_TOP_DIR,env_i)
     
                  gripper_name = 'None'
                  if int(gripper_id) == 12:
                    gripper_ends_with = '_par_bh282tmp_label_stage2.npy'
                    gripper_name = 'bh_282'
                  elif int(gripper_id) == 11:
                    gripper_ends_with = '_par_robotiq_3f_fullest_tmp_label_stage2.npy'
                    gripper_name = 'robotiq_3f'
                  elif int(gripper_id) == 13:
                    gripper_ends_with = '_par_kinova_3f_fullest_tmp_labelkinova_stage2.npy'
                    gripper_name = 'kinova_kg3'
  
                  tmp_label_dirs = [line for line in os.listdir(tmp_label_dir) if line.endswith(gripper_ends_with)]
                  tmp_label_dir = os.path.join(tmp_label_dir, tmp_label_dirs[0]) 
                  tmp_label = np.load(tmp_label_dir)
                  tmp_label_dir_w = tmp_label_dir.split('stage2.npy')[0] + 'stage2_w.npy'
                  tmp_label_w = np.load(tmp_label_dir_w)
                  top1024_nor = in_objnor[bi][out_single_point_top_1024_index[bi]]

                  for k in range(TOP_K):
                      gt_label[k][out_single_index_stage2_tmp[:,0]] = tmp_label[old_id_new_list[bi][out_single_point_top_index[bi][k]],out_single_index_stage2_tmp[:,1]]
                      gt_label_w[k][out_single_index_stage2_tmp[:,0]] = tmp_label_w[old_id_new_list[bi][out_single_point_top_index[bi][k]],out_single_index_stage2_tmp[:,1]]
                      single_nor = in_objnor[bi][out_single_point_top_index[bi][k]]
                      nor_diff = np.sum(top1024_nor * single_nor,axis=-1)
                      nor_diff_flag = np.where(nor_diff > -0.7)[0]
                      gt_label[k][nor_diff_flag] = 0
                      gt_label_w[k][nor_diff_flag] = 0
                  gt_label_w = gt_label_w / (np.max(gt_label_w)/2.0 + 1e-5) + 0.5

              gt_two_points_label_list.append(gt_label)
              gt_two_points_label_w_list.append(gt_label_w)

        gt_two_points_label = np.array(gt_two_points_label_list).astype(np.int32)
        gt_two_points_label = np.reshape(gt_two_points_label,[-1,TOP_K * TOP_K2])
        gt_two_points_label_w = np.array(gt_two_points_label_w_list)
        gt_two_points_label_w = np.reshape(gt_two_points_label_w,[-1,TOP_K * TOP_K2])

        #print(np.sum(gt_two_points_label))
        out_single_point_top_1024_index_v2, out_two_points_top_index, loss2 = sess.run([out_single_point_top_1024_index_tf, out_two_points_top_index_tf, loss_stage2],feed_dict={gt_two_points_label_w_tf: gt_two_points_label_w, gq_label_tf:stage1_gt_gq, gt_pcn_tf:in_objnor, gripper_feat_tf: in_gripper_feat, obj_pc_tf: in_objenv, gt_two_points_label_tf: gt_two_points_label})
        #out_single_point_top_1024_index_v2, out_two_points_top_index, loss2, _ = sess.run([out_single_point_top_1024_index_tf, out_two_points_top_index_tf, loss_stage2, train_op_stage2],feed_dict={gq_label_tf:stage1_gt_gq, gt_pcn_tf:in_objnor, gripper_feat_tf: in_gripper_feat, obj_pc_tf: in_objenv, gt_two_points_label_tf: gt_two_points_label,gt_two_points_label_w_tf: gt_two_points_label_w})
        train_loss_stage2 += loss2

        assert np.all(out_single_point_top_1024_index_v2 == out_single_point_top_1024_index)    

        for gj in range(FLAGS.batch_size):
          two_points_label =  out_two_points_top_index[gj]

          top_two_points_index_1 = two_points_label // TOP_K2
          top_two_points_index_1 = out_single_point_top_index[gj][top_two_points_index_1]
          top_two_points_index_2 = out_single_point_top_1024_index[gj][two_points_label % TOP_K2]
   
          top1_two_points_index_1 = top_two_points_index_1[0:24]
          top1_two_points_index_2 = top_two_points_index_2[0:24]

          acc_map = gt_two_points_label[gj][two_points_label]
          weight_map = gt_two_points_label_w[gj][two_points_label]

          if np.sum(acc_map[0:1]) > 0:
            stage2_acc_top1 += 1.0
          else:
            p1 = in_objenv[gj][top1_two_points_index_1[0:1]]
            p2 = in_objenv[gj][top1_two_points_index_2[0:1]]
            stree = KDTree(in_objenv[gj])
            ind1 = stree.query_radius(p1,r=0.005)[0]
            ind2 = stree.query_radius(p2,r=0.005)[0]
            nind1 = old_id_new_list[gj][ind1]
            nind2 = old_id_new_list[gj][ind2]
            iflag = 0
            for nd1 in nind1:
              for nd2 in nind2:
                if tmp_label[nd1,nd2]:
                  iflag = 1
                  break 
            stage2_acc_top1 += iflag
               
          if np.sum(acc_map[0:10]) > 0:
            stage2_acc_top10 += 1.0
          else:
            for ppp in range(10):
              p1 = in_objenv[gj][top1_two_points_index_1[ppp:ppp+1]]
              p2 = in_objenv[gj][top1_two_points_index_2[ppp:ppp+1]]
              stree = KDTree(in_objenv[gj])
              ind1 = stree.query_radius(p1,r=0.005)[0]
              ind2 = stree.query_radius(p2,r=0.005)[0]
              nind1 = old_id_new_list[gj][ind1]
              nind2 = old_id_new_list[gj][ind2]
              iflag = 0
              for nd1 in nind1:
                for nd2 in nind2:
                  if tmp_label[nd1,nd2]:
                    iflag = 1
                    break 
              if iflag:
                stage2_acc_top10 += 1.0
                break

          if 1:
            neighbor = 0.0
            p1 = in_objenv[gj][top1_two_points_index_1[0:1]]
            p2 = in_objenv[gj][top1_two_points_index_2[0:1]]
            stree = KDTree(in_objenv[gj])
            ind1 = stree.query_radius(p1,r=0.005)[0]
            ind2 = stree.query_radius(p2,r=0.005)[0]
            nind1 = old_id_new_list[gj][ind1]
            nind2 = old_id_new_list[gj][ind2]
            iflag = 0
            for nd1 in nind1:
              for nd2 in nind2:
                if tmp_label[nd1,nd2]:
                  neighbor += 1.0
            final_neighbor += neighbor / float(len(nind1) * len(nind2))
            #print(neighbor, float(len(nind1) * len(nind2)), neighbor / float(len(nind1) * len(nind2)))

          gt_two_points_label_pos = np.where(gt_two_points_label[gj] > 0)[0]

          gt_stage2_acc_top1 +=  (np.sum(gt_two_points_label[gj][gt_two_points_label_pos[0:TOP_K]]))/float(TOP_K)#acc_map[0]
 
          gt_top_two_points_index_1 = gt_two_points_label_pos // TOP_K2	
          gt_top_two_points_index_1 = out_single_point_top_index[gj][gt_top_two_points_index_1]
          gt_top_two_points_index_2 = out_single_point_top_1024_index[gj][gt_two_points_label_pos % TOP_K2]

          gt_top1_two_points_index_1 = gt_top_two_points_index_1[0:24]
          gt_top1_two_points_index_2 = gt_top_two_points_index_2[0:24]

        for gj in range(FLAGS.batch_size):
          two_points_label =  out_two_points_top_index[gj]

          top_two_points_index_1 = two_points_label // TOP_K2
          top_two_points_index_1 = out_single_point_top_index[gj][top_two_points_index_1]
          top_two_points_index_2 = out_single_point_top_1024_index[gj][two_points_label % TOP_K2]
   
          top1_two_points_index_1 = top_two_points_index_1[0:12]
          top1_two_points_index_2 = top_two_points_index_2[0:12]

          gt_two_points_label_pos = np.where(gt_two_points_label[gj] > 0)[0]

          gt_top_two_points_index_1 = gt_two_points_label_pos // TOP_K2
          gt_top_two_points_index_1 = out_single_point_top_index[gj][gt_top_two_points_index_1]
          gt_top_two_points_index_2 = out_single_point_top_1024_index[gj][gt_two_points_label_pos % TOP_K2]

          gt_top1_two_points_index_1 = gt_top_two_points_index_1[0:250]
          gt_top1_two_points_index_2 = gt_top_two_points_index_2[0:250]

          gtt = np.vstack([[top1_two_points_index_1],[top1_two_points_index_2]]).T[:50]
          if 1:
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
        
      # stage3 oldr
      if 0:
        gt_corr_label_list = []
  
        for bbi in range(int(FLAGS.batch_size/gripper_size)):
          for ggi in range(gripper_size):
            gripper_id = gripper_index[ggi]
            bi = bbi * gripper_size + ggi
            env_i = str(train_val_test_list._train[bt_index[bbi]])
            env_dir = os.path.join(DATA_TOP_DIR, env_i)
            gripper_path = GRIPPER_TOP_DIR


            if gripper_id > 10:
              gripper_name = 'None'
              if int(gripper_id) == 12:
                 gripper_ends_with = '_par_bh282.npy'
                 gripper_name = 'bh_282'
              elif int(gripper_id) == 11:
                 gripper_ends_with = '_robotiq_3f_fullest_v3.npy'
                 gripper_name = 'robotiq_3f'
              elif int(gripper_id) == 13:
                 gripper_ends_with = '_kinova_3f_fullest_v3.npy'
                 gripper_name = 'kinova_kg3'
              gt_3fgripper_path_dirs = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith(gripper_ends_with)]
              if 1:            
                gt_robotiq3f_path = gt_3fgripper_path_dirs[0]
                gt_label = np.zeros((TOP_K, TOP_K2))

                gt_3f_label = np.load(gt_robotiq3f_path)
                tmp_label = np.zeros((2048,2048,2048),dtype='bool') 
                if 1:
                  tmp_label[gt_3f_label[:,0].astype(np.int32),gt_3f_label[:,1].astype(np.int32),gt_3f_label[:,2].astype(np.int32)] = 1 
                  tmp_label[gt_3f_label[:,0].astype(np.int32),gt_3f_label[:,2].astype(np.int32),gt_3f_label[:,1].astype(np.int32)] = 1 
                  tmp_label[gt_3f_label[:,1].astype(np.int32),gt_3f_label[:,0].astype(np.int32),gt_3f_label[:,2].astype(np.int32)] = 1 
                  tmp_label[gt_3f_label[:,1].astype(np.int32),gt_3f_label[:,2].astype(np.int32),gt_3f_label[:,0].astype(np.int32)] = 1 
                  tmp_label[gt_3f_label[:,2].astype(np.int32),gt_3f_label[:,0].astype(np.int32),gt_3f_label[:,1].astype(np.int32)] = 1 
                  tmp_label[gt_3f_label[:,2].astype(np.int32),gt_3f_label[:,1].astype(np.int32),gt_3f_label[:,0].astype(np.int32)] = 1 

                  if int(gripper_id) == 12:
                    gt_3fgripper_path_dirs = [os.path.join(env_dir,f) for f in os.listdir(env_dir) if f.endswith('_robotiq_3f_fullest_v3.npy')]
                    gt_robotiq3f_path = gt_3fgripper_path_dirs[0]
                    gt_3f_label_v2 = np.load(gt_robotiq3f_path)
                
                    tmp_label[gt_3f_label_v2[:,0].astype(np.int32),gt_3f_label_v2[:,1].astype(np.int32),gt_3f_label_v2[:,2].astype(np.int32)] = 1 
                    tmp_label[gt_3f_label_v2[:,0].astype(np.int32),gt_3f_label_v2[:,2].astype(np.int32),gt_3f_label_v2[:,1].astype(np.int32)] = 1 
                    tmp_label[gt_3f_label_v2[:,1].astype(np.int32),gt_3f_label_v2[:,0].astype(np.int32),gt_3f_label_v2[:,2].astype(np.int32)] = 1 
                    tmp_label[gt_3f_label_v2[:,1].astype(np.int32),gt_3f_label_v2[:,2].astype(np.int32),gt_3f_label_v2[:,0].astype(np.int32)] = 1 
                    tmp_label[gt_3f_label_v2[:,2].astype(np.int32),gt_3f_label_v2[:,0].astype(np.int32),gt_3f_label_v2[:,1].astype(np.int32)] = 1 
                    tmp_label[gt_3f_label_v2[:,2].astype(np.int32),gt_3f_label_v2[:,1].astype(np.int32),gt_3f_label_v2[:,0].astype(np.int32)] = 1 

                   
                  out_single_index_stage2 = [(idx, gt_env_id_list[bi][l]) for idx, l in enumerate(out_single_point_top_1024_index[bi]) if l < gt_env_num_list[bi]]
                  out_single_index_stage2_tmp = np.array(out_single_index_stage2)
                  
                  two_points_label =  out_two_points_top_index[bi]
                  top_two_points_index_1 = two_points_label // TOP_K2
                  top_two_points_index_1 = out_single_point_top_index[bi][top_two_points_index_1]
                  top_two_points_index_2 = out_single_point_top_1024_index[bi][two_points_label % TOP_K2]
                  step2 = time.time()
                  for k in range(TOP_K):
                    if top_two_points_index_1[k] < gt_env_num_list[bi] and top_two_points_index_2[k] < gt_env_num_list[bi]:
                      gt_label[k][out_single_index_stage2_tmp[:,0]] = tmp_label[gt_env_id_list[bi][top_two_points_index_1[k]],  gt_env_id_list[bi][top_two_points_index_2[k]], out_single_index_stage2_tmp[:,1]]
                  step3= time.time()
              gt_corr_label_list.append(gt_label)

        gt_corr_label = np.array(gt_corr_label_list).astype(np.int32)
        gt_corr_label = np.reshape(gt_corr_label,[-1,TOP_K * TOP_K2])


        if 1:
         for bbi in range(int(FLAGS.batch_size/gripper_size)):
          for ggi in range(gripper_size):
            gripper_id = gripper_index[ggi]
            bi = bbi * gripper_size + ggi
            env_i = str(train_val_test_list._train[bt_index[bbi]])
    
            env_save_dir = os.path.join(DATA_TOP_DIR,env_i)
            if not os.path.exists(env_save_dir):
              os.makedirs(env_save_dir)
            s1_top1024_dir = os.path.join(env_save_dir,str(gripper_id)+'_s1_top1024.npy')     
            np.save(s1_top1024_dir,out_single_point_top_1024_index[gj])           
            s1_topk_dir = os.path.join(env_save_dir,str(gripper_id)+'_s1_topk.npy') 
            np.save(s1_topk_dir,out_single_point_top_index[gj])   
            s2_topk2_dir = os.path.join(env_save_dir,str(gripper_id)+'_s2_top1024.npy')  
            np.save(s2_topk2_dir,out_two_points_top_index[gj])  
            s3_gt_dir = os.path.join(env_save_dir,str(gripper_id)+'_s3_gt.npy')
            np.save(s3_gt_dir,gt_corr_label[gj])

        out_corr_top_index_stage3, loss3 = sess.run([out_corr_top_index_stage3_tf, loss_stage3],feed_dict={gripper_feat_tf: in_gripper_feat, obj_pc_tf: in_objenv, gt_corr_label_stage3_tf: gt_corr_label})
        
        step4=time.time()
        
        if 1:
         for gj in range(FLAGS.batch_size):
          third_point_set_label = out_corr_top_index_stage3[gj]

          top_corr2_index = third_point_set_label // TOP_K2
       
          two_points_label_ = out_two_points_top_index[gj] 
          top_two_points_index_1_ = two_points_label_ // TOP_K2
          top_two_points_index_1_ = out_single_point_top_index[gj][top_two_points_index_1_]
          top_two_points_index_2_ = out_single_point_top_1024_index[gj][two_points_label_ % TOP_K2]
          
          top_f1_index_1 = top_two_points_index_1_[top_corr2_index]
          top_f2_index_2 = top_two_points_index_2_[top_corr2_index]
          top_f3_index_3 = out_single_point_top_1024_index[gj][third_point_set_label % TOP_K2]
     
          top1_f1_index_1 = top_f1_index_1[0:10]
          top1_f2_index_2 = top_f2_index_2[0:10]
          top1_f3_index_3 = top_f3_index_3[0:10]

          gt_three_points_label = np.where(gt_corr_label[gj] > 0)[0]
          num_gt_three = len(gt_three_points_label)
          
          gt_top_corr2_index = gt_three_points_label // TOP_K2
        
          gt_top_f1_index_1_ = top_two_points_index_1_[gt_top_corr2_index]
          gt_top_f2_index_2_ = top_two_points_index_2_[gt_top_corr2_index]
          gt_top_f3_index_3_ = out_single_point_top_1024_index[gj][gt_three_points_label % TOP_K2]
  
          gt_top1_f1_index_1 = gt_top_f1_index_1_[0:1]
          gt_top1_f2_index_2 = gt_top_f2_index_2_[0:1]
          gt_top1_f3_index_3 = gt_top_f3_index_3_[0:1]

          ####### calculate neighbor
          p1 = in_objenv[gj][top1_f1_index_1[0:1]]
          p2 = in_objenv[gj][top1_f2_index_2[0:1]]
          p3 = in_objenv[gj][top1_f3_index_3[0:1]]
          stree = KDTree(in_objenv[gj])
          ind1 = stree.query_radius(p1,r=0.005)[0]
          ind2 = stree.query_radius(p2,r=0.005)[0]
          ind3 = stree.query_radius(p3,r=0.005)[0]
          nind1 = old_id_new_list[gj][ind1]
          nind2 = old_id_new_list[gj][ind2]
          nind3 = old_id_new_list[gj][ind3]
          neighbor = 0
          flag_top1 = 0
          for nd1 in nind1:
            for nd2 in nind2:
              for nd3 in nind3:
                if tmp_label[nd1,nd2,nd3]:
                  neighbor += 1
                  flag_top1 = 1
          final_neighbor += neighbor / float(len(nind1) * len(nind2) * len(nind3))
          print(neighbor,float(len(nind1) * len(nind2) * len(nind3)), neighbor / float(len(nind1) * len(nind2) * len(nind3)))
          if gt_corr_label[gj][third_point_set_label[0:1]] > 0:
            stage3_acc_top1 += 1.0
          else:
            p1 = in_objenv[gj][top1_f1_index_1[0:1]]
            p2 = in_objenv[gj][top1_f2_index_2[0:1]]
            p3 = in_objenv[gj][top1_f3_index_3[0:1]]
            stree = KDTree(in_objenv[gj])
            ind1 = stree.query_radius(p1,r=0.005)[0]
            ind2 = stree.query_radius(p2,r=0.005)[0]
            ind3 = stree.query_radius(p3,r=0.005)[0]
            nind1 = old_id_new_list[gj][ind1]
            nind2 = old_id_new_list[gj][ind2]
            nind3 = old_id_new_list[gj][ind3]
            iflag = 0
            for nd1 in nind1:
              for nd2 in nind2:
                for nd3 in nind3:
                  if tmp_label[nd1,nd2,nd3]:
                    iflag = 1
                    break
            stage3_acc_top1 += iflag


          if np.sum(gt_corr_label[gj][third_point_set_label[0:10]]) > 0:
            stage3_acc_top10 += 1.0
          elif flag_top1 > 0:
            stage3_acc_top10 += 1.0
          else:
            for ppp in range(10):
              p1 = in_objenv[gj][top1_f1_index_1[ppp:ppp+1]]
              p2 = in_objenv[gj][top1_f2_index_2[ppp:ppp+1]]
              p3 = in_objenv[gj][top1_f3_index_3[ppp:ppp+1]]
              stree = KDTree(in_objenv[gj])
              ind1 = stree.query_radius(p1,r=0.005)[0]
              ind2 = stree.query_radius(p2,r=0.005)[0]
              ind3 = stree.query_radius(p3,r=0.005)[0]
              nind1 = old_id_new_list[gj][ind1]
              nind2 = old_id_new_list[gj][ind2]
              nind3 = old_id_new_list[gj][ind3]
              iflag = 0
              for nd1 in nind1:
                for nd2 in nind2:
                  for nd3 in nind3:
                    if tmp_label[nd1,nd2,nd3]:
                      iflag = 1
                      break
              if iflag:
                break
            stage3_acc_top10 += iflag

 
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

          gt_three_points_label = np.where(gt_corr_label[gj] > 0)[0]
          num_gt_three = min(len(gt_three_points_label),12)
          gt_top_corr2_index = gt_three_points_label // TOP_K2

          gt_top_f1_index_1 = top_two_points_index_1_[gt_top_corr2_index]
          gt_top_f2_index_2 = top_two_points_index_2_[gt_top_corr2_index]
          gt_top_f3_index_3 = out_single_point_top_1024_index[gj][gt_three_points_label % TOP_K2]

          gt_top1_f1_index_1 = gt_top_f1_index_1[0:num_gt_three]
          gt_top1_f2_index_2 = gt_top_f2_index_2[0:num_gt_three]
          gt_top1_f3_index_3 = gt_top_f3_index_3[0:num_gt_three]


         if 1:
          s_p = np.copy(in_objenv[0])
          s_p[:,2] *= -1.0
          c_pred = np.zeros((2048,3))
          c_pred[:,0] = 255.0#255#194.0
          c_pred[:,1] = 255#194.0
          c_pred[:,2] = 255.0#214.0
          c_pred[gt_top1_f1_index_1,0] = 0.0
          c_pred[gt_top1_f1_index_1,1] = 255.0
          c_pred[gt_top1_f1_index_1,2] = 0.0
          c_pred[gt_top1_f2_index_2,0] = 255.0
          c_pred[gt_top1_f2_index_2,1] = 0.0
          c_pred[gt_top1_f2_index_2,2] = 0.0
          c_pred[gt_top1_f3_index_3,0] = 0.0
          c_pred[gt_top1_f3_index_3,1] = 0.0
          c_pred[gt_top1_f3_index_3,2] = 255.0
          showpoints(s_p,c_gt=c_pred,waittime=5,ballradius=4,freezerot=False) ## GRB
          #input("r")
    train_loss_stage2 /= float(num_batch)
    train_loss /= float(num_batch)
    train_loss_stage3 /= float(num_batch)
    train_loss_nor /= float(num_batch)
    score_stage2 /= float(num_batch)
    stage1_acc_top1 /= (float(num_batch)*FLAGS.batch_size)
    stage1_acc_top10 /= (float(num_batch)*FLAGS.batch_size)
    stage2_acc_top1 /= (float(num_batch)*FLAGS.batch_size)
    stage2_acc_top10 /= (float(num_batch)*FLAGS.batch_size)
    stage3_acc_top1 /= (float(num_batch)*FLAGS.batch_size)
    stage3_acc_top10 /= (float(num_batch)*FLAGS.batch_size)
    final_neighbor /= (float(num_batch)*FLAGS.batch_size)
 
    print("epoch loss %d : %f loss stage2:%f  loss stage3:%f loss nor:%f",epoch, train_loss,train_loss_stage2,train_loss_stage3, train_loss_nor)
    print("stage1_top1_acc", stage1_acc_top1, " stage1_top30_acc:", stage1_acc_top10)
    print("stage2_top1_acc", stage2_acc_top1, " stage2_top10_acc:", stage2_acc_top10)
    print("stage3_top1_acc", stage3_acc_top1, " stage3_top10_acc:", stage3_acc_top10)
    print("final_neighbor", final_neighbor)


if __name__ == "__main__":
  # Fist point training ends at 223 $$$ 435
  #for i in range(223,440):
  #  restore_stage3_v2(i)
  #  test(i)
  #test(1)
  restore_stage3(220)
  test(0)
  #for i in range(2115,2530):
  #  restore_stage3(i)
  #  test(2050)
