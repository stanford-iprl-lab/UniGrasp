from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys
import argparse
import matplotlib.pyplot as plt
import tflearn

from tf_models.gripper_auto_encoding import pc_encoder as pc_encoder
from tf_models.gripper_auto_encoding import pc_decoder

import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR,'../')

# Basic model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--saver_dir',default='./saved_models/',help='Directory to save the trained model')
parser.add_argument('--learning_rate',type=float,default=0.0005,help='Initial learning rate')
parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
parser.add_argument('--batch_size',type=int,default=2,help='Number of examples within a batch')
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
TOP_K = 256 


#DATA_TOP_DIR = os.path.join(ROOT_DIR,'Data','BlensorResult')
DATA_TOP_DIR = os.path.join(ROOT_DIR,'Data','Gripper','Data_noR')
#GRIPPER_TOP_DIR = os.path.join(ROOT_DIR,'Data','Gripper_DB')

in_gripper_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_in')
gt_gripper_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_gt')


with tf.variable_scope('gripper_encoder'):
  gripper_feat_tf = pc_encoder(in_gripper_tf)

with tf.variable_scope('gripper_decoder'):
  out_gripper_tf = pc_decoder(gripper_feat_tf)
  #reta,retb,retc,retd=nn_distance(out_gripper_tf, gt_gripper_tf)
  #loss = tf.reduce_sum(reta) + tf.reduce_sum(retc) 
  #train_op = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init_op)

SAVER = tf.train.Saver(max_to_keep=1000)


def save_model(epoch):
  save_top_dir = os.path.join('./saved_models',"gripper_v2")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  if epoch == 0:
    SAVER.save(sess, ckpt_path, write_meta_graph=True)
  else:
    SAVER.save(sess, ckpt_path, write_meta_graph=False)
  print("Saving model at epoch %d to %s" % (epoch, ckpt_path))


def restore(epoch):
  save_top_dir = os.path.join('./saved_models',"gripper_v2")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)
  #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  #print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name="", all_tensors=False, all_tensor_names=True)


def rpy_rotmat(rpy):
  rotmat = np.zeros((3,3))
  roll   = rpy[0]
  pitch  = rpy[1]
  yaw    = rpy[2]
  rotmat[0,0] = np.cos(yaw) * np.cos(pitch)
  rotmat[0,1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
  rotmat[0,2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
  rotmat[1,0] = np.sin(yaw) * np.cos(pitch)
  rotmat[1,1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
  rotmat[1,2] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
  rotmat[2,0] = - np.sin(pitch)
  rotmat[2,1] = np.cos(pitch) * np.sin(roll)
  rotmat[2,2] = np.cos(pitch) * np.cos(roll)
  return rotmat


def test(base=0):
  #num_examples = len(train_val_test_list._train)
  #print("num training examples %d" % (num_examples))
  #num_batch = int(num_examples/FLAGS.batch_size)
  #print("num_training_batch", num_batch)
  #two_env_i = ['kinova_kg3_0.npy','kinova_kg3_1.npy']
 
  DATA_TOP_DIR = '/scr2/MetaGrasp/Data/Gripper/Data_DB/allegro' 
  in_gripper_file_list = [line for line in os.listdir(DATA_TOP_DIR)]
  in_gripper_list = []
  for idx, env_i in enumerate(in_gripper_file_list):
      env_dir = os.path.join(DATA_TOP_DIR, env_i)
      obj_pcs = np.load(env_dir)
      in_gripper_list.append(obj_pcs)
      #if idx % 20 == 19:
      in_gripper = np.array(in_gripper_list)   
      gt_gripper = in_gripper
      out_gripper, gripper_feat = sess.run([out_gripper_tf, gripper_feat_tf],feed_dict={in_gripper_tf: in_gripper, gt_gripper_tf:gt_gripper})

      out_dir = '/scr2/MetaGrasp/Data/Gripper/Feat'
      out_dir_file = os.path.join(out_dir,env_i)
      np.save(out_dir_file, gripper_feat)
      in_gripper_list = []
  print(gripper_feat.shape)
  gripper_mean = np.mean(gripper_feat,axis=0)
  gripper_max =  np.max(gripper_feat,axis=0)
  gripper_min = np.min(gripper_feat,axis=0)
  print(gripper_mean.shape)
  recon_dir = DATA_TOP_DIR

  mean_feat_file = os.path.join(recon_dir,'mean.npy')
  max_feat_file = os.path.join(recon_dir,'max.npy')
  min_feat_file = os.path.join(recon_dir,'min.npy')
  
  print(mean_feat_file)
  print(max_feat_file)
  print(min_feat_file)
  np.save(mean_feat_file,gripper_mean)
  np.save(max_feat_file,gripper_max)
  np.save(min_feat_file,gripper_min)

  #for bi in range(FLAGS.batch_size):
  #      env_i = two_env_i[bi]#str(train_val_test_list._train[bt_index[bi]])      
  #      recon_dir_file = os.path.join(recon_dir,env_i)
  #      print(recon_dir_file)
        #np.save(recon_dir_file, out_gripper[bi])

  #for bi in range(FLAGS.batch_size):
  #  middle_dir = os.path.join(recon_dir,'middle'+str(bi)+'.npy')
    #np.save(middle_dir,out_gripper_new[bi])

  if 1:
       for gj in range(1):
        green = np.zeros((4096,3))
        green[:2048,0] = 255.0
        green[2048:,1] = 255.0
        pred_gripper = np.copy(out_gripper_new[gj])
        #tmp_gripper = pred_gripper[:,0]
        #pred_gripper[:,0] = pred_gripper[:,2]
        #pred_gripper[:,2] = tmp_gripper

        gt__gripper = np.copy(gt_gripper[gj])
        #tmp__gripper = gt__gripper[:,0]
        #gt__gripper[:,0] = gt__gripper[:,2]
        #gt__gripper[:,2] = tmp__gripper

        gripper_two = np.zeros((4096,3))
        gripper_two[:2048,:] = pred_gripper
        gripper_two[2048:,:] = gt__gripper
        showpoints(gripper_two,c_gt=green,waittime=50,freezerot=False) ### GRB
        input("gripper")


if __name__ == "__main__":
  save_model(0)
  for i in range(2248,2250,2):#3000,2):
    restore(i)
    test(i)
