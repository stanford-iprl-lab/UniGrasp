from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys
import argparse

from tf_models.gripper_auto_encoding import pc_encoder as pc_encoder
from tf_models.gripper_auto_encoding import pc_decoder

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

sys.path.insert(0,"../vis_3d")
from show3d_balls import showpoints

if not os.path.exists(FLAGS.saver_dir):
  os.mkdir(FLAGS.saver_dir)

if not os.path.exists(FLAGS.log_dir):
  os.mkdir(FLAGS.log_dir)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

in_gripper_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_in')
gt_gripper_tf = tf.placeholder(tf.float32,[None,2048,3],'gripper_gt')


with tf.variable_scope('gripper_encoder'):
  gripper_feat_tf = pc_encoder(in_gripper_tf)

with tf.variable_scope('gripper_decoder'):
  out_gripper_tf = pc_decoder(gripper_feat_tf)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init_op)

SAVER = tf.train.Saver(max_to_keep=1000)

def restore(epoch):
  save_top_dir = os.path.join("../saved_models/gripper_representation")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)


def test(gripper_dir=None,gripper_name="robotiq_3f"):
  in_gripper_file_list = [line for line in os.listdir(gripper_dir) if line.startswith(gripper_name)]
  in_gripper_list = []
  for idx, env_i in enumerate(in_gripper_file_list):
      env_dir = os.path.join(gripper_dir, env_i)
      obj_pcs = np.load(env_dir)
      in_gripper_list.append(obj_pcs)

  in_gripper = np.array(in_gripper_list)   
  gt_gripper = in_gripper
  out_gripper, gripper_feat = sess.run([out_gripper_tf, gripper_feat_tf],feed_dict={in_gripper_tf: in_gripper, gt_gripper_tf:gt_gripper})

  print(gripper_feat.shape)
  gripper_mean = np.mean(gripper_feat,axis=0)
  gripper_max =  np.max(gripper_feat,axis=0)
  gripper_min = np.min(gripper_feat,axis=0)
  print(gripper_mean.shape)

  recon_dir = gripper_dir
  mean_feat_file = os.path.join(recon_dir,'mean.npy')
  max_feat_file = os.path.join(recon_dir,'max.npy')
  min_feat_file = os.path.join(recon_dir,'min.npy')

  print(mean_feat_file)
  print(max_feat_file)
  print(min_feat_file)
  np.save(mean_feat_file,gripper_mean)
  np.save(max_feat_file,gripper_max)
  np.save(min_feat_file,gripper_min)


  if 1:
       for gj in range(len(in_gripper)):
        green = np.zeros((4096,3))
        green[:2048,0] = 255.0
        green[2048:,1] = 255.0
        pred_gripper = np.copy(out_gripper[gj])
        
        gt__gripper = np.copy(gt_gripper[gj])
        gripper_two = np.zeros((4096,3))
        gripper_two[:2048,:] = pred_gripper
        gripper_two[2048:,:] = gt__gripper
        showpoints(gripper_two,c_gt=green,waittime=50,freezerot=False) ### GRB
        #input("gripper")


if __name__ == "__main__":
  #### restore the model of auto encoder
  restore(2248)

  #### specify the folder path of point clouds of the gripper
  gripper_dir = "../data/grippers/robotiq_3f"
  gripper_name = "robotiq_3f"
  test(gripper_dir,gripper_name)
