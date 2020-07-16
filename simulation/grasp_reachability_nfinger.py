import sys
import os
import numpy as np
import math
from math import cos, sin
import scipy.ndimage

from mayavi import mlab as mayalab
from urdf_parser_py.urdf import URDF

def plot_pc(pcs,color=None,scale_factor=.05,mode='point'):
  if color == 'r':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(1,0,0))
  elif color == 'blue':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,0,1))
  elif color == 'g':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,0))
  elif color == 'ycan':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,1))
  else:
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor)

def plot_pc_with_normal(pcs,pcs_n):
  mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], mode='arrow')

from pgm_loader import read_pgm_xyz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from geolib.objfile import OBJ
from geolib.cuboid import Cuboid
from geolib.cone import Cone
from math_utils import quaternionProduct,camRotQuaternion,camPosToQuaternion,obj_centened_camera_pos,quaternion_matrix,tran_rot

np.set_printoptions(precision=4,suppress=True,linewidth=300)

class GraspConfig:
  def __init__(self,graspmap_filepath, obj_top_dir, friction_coef):
    self.friction_coef = friction_coef
    self.obj_top_dir = obj_top_dir

    self.graspmap_filepath = graspmap_filepath

    self.grasp_para_savepath = graspmap_filepath.split('.pgm')[0] + '_par.npz'
    self.pcn_para_savepath = graspmap_filepath.split('.pgm')[0] + '_pcn.npz'
    self.tran_rot_filepath = graspmap_filepath.split('00000.pgm')[0] + '_matrix_wolrd.txt'
    self.good_grasp = graspmap_filepath.split('.pgm')[0]+'_par_robotiq_3f.npz'
    self.grasp_config_file = graspmap_filepath.split('.pgm')[0]+'_par_3f_config.npy'
    self.grasp_index_file =  graspmap_filepath.split('.pgm')[0]+'_par_3f_512.npz'

    self.grasp_config = np.load(self.grasp_config_file)
    self.grasp_score = self.grasp_config[:,3]
    self.grasp_bestId = np.argmax(self.grasp_score)
    self.grasp_bestIdList = np.where(self.grasp_score > 0.0001)[0]
    self.contacts1 = self.grasp_config[:,0].astype(np.int32)
    self.contacts2 = self.grasp_config[:,1].astype(np.int32)
    self.contacts3 = self.grasp_config[:,2].astype(np.int32)

    self.grasp_index = np.load(self.grasp_index_file)
 
    self.grasp_xyz  = read_pgm_xyz(self.graspmap_filepath)

    self.model_id =  self.graspmap_filepath.split('_rho')[0].split('/')[-1]
    tmp = self.graspmap_filepath.split('.pgm')[0].split('_')

    self.table = np.zeros((4,3))
    self.table[0] = np.array([-0.1,0.0,-0.01])
    self.table[1] = np.array([0.1,0.0,-0.01])
    self.table[2] = np.array([0.0,-0.1,-0.01])
    self.table[3] = np.array([0.0,0.1,-0.01])

    x_cam = float(tmp[-7].split('xcam')[1])
    y_cam = float(tmp[-6].split('ycam')[1])
    z_cam = float(tmp[-5].split('zcam')[1])

    x_dim = float(tmp[-3].split('xdim')[1])
    z_dim = float(tmp[-1].split('zdim')[1])
    y_dim = float(tmp[-2].split('ydim')[1])

    self.scale = float(tmp[-4].split('scale')[1])

    rho =  float(tmp[-11].split('rho')[1])

    ##### read all the necessary parameters    
    self.height,self.width,self.depth = self.grasp_xyz.shape
    self.grasp_xyz = self.grasp_xyz.reshape((-1,3)) # point clouds are in camera coordinates

    azimuth_deg = float(tmp[-10].split('azi')[1])
    elevation_deg = float(tmp[-9].split('ele')[1])
    theta_deg = float(tmp[-8].split('theta')[1])

    self.azimuth_deg = azimuth_deg

    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx , cy , cz)
    q2 = camRotQuaternion(cx, cy , cz, theta_deg)
    q = quaternionProduct(q2, q1)
    R = quaternion_matrix(q)[0:3,0:3]

    ####  Transform point cloud from camera coordinate to blensor coordinate 
    self.grasp_xyz = self.grasp_xyz - np.array([x_cam,y_cam,z_cam])
    self.grasp_xyz[:,2] *= -1.0
    self.grasp_xyz = R.dot(self.grasp_xyz.T).T
    self.table[:,2] = np.min(self.grasp_xyz[:,2])
    #plot_pc(self.table,scale_factor=0.005,mode='sphere',color='r')

    ####
    tran, rot = tran_rot(self.tran_rot_filepath)

    ###### pc_test
    self.pcn = np.load(self.pcn_para_savepath)['pcn']
    self.pc_test = self.pcn[:,0:3]
    self.pc_normal_test = self.pcn[:,3:6]
    self.center = np.zeros((1,3))
    self.center = np.copy(tran)
    self.center = self.center.reshape((-1,3))
    plot_pc(self.center,color='r',mode='sphere',scale_factor=0.01)
    self.pcn = np.hstack([self.pc_test, self.pc_normal_test])
   #np.savez(self.pcn_para_savepath,pcn=self.pcn)



def robotiq_3f_pinch_grasp_checker(p1,p2,p3,n1,n2,n3):
  if np.inner(n1,n2) > 0.8 or np.inner(n1,n3) > 0.8 or np.inner(n2,n3) > 0.8:
    normal = np.cross((p1-p2),(p1-p3))
    n = normal/(np.linalg.norm(normal) + 1e-16)
    c = (p1 + p2 + p3) / 3.0
    if np.abs(n[2]) > 0.5:  
      if np.abs(np.inner(n,n1)) < 0.3 and np.abs(np.inner(n,n2)) < 0.3 and np.abs(np.inner(n,n3)) < 0.3: 
        p1c = p1 - c
        p1c = p1c / (np.linalg.norm(p1c) + 1e-16)
        p2c = p2 - c
        p2c = p2c / (np.linalg.norm(p2c) + 1e-16)
        p3c = p3 - c
        p3c = p3c / (np.linalg.norm(p3c) + 1e-16)
        if np.inner(p1c,n1) > 0.1 and np.inner(p2c,n2) > 0.1 and np.inner(p3c,n3) > 0.1:
          return True
        else:
          return False
      else:
        return False
    else:
      return False
  else:
    return False


def robotiq_3f_initial_frame(p1,p2,p3,n1,n2,n3):
  normal = np.cross((p1-p2),(p1-p3))
  n = normal/(np.linalg.norm(normal) + 1e-16)
  c = (p1 + p2 + p3) / 3.0
  gripper_frame = np.eye(4)
  if n[2] > 0.05:
    gripper_center_1 = (c + n * 0.18)
    gripper_frame[:3,2] = -n
    gripper_frame[:3,3] = gripper_center_1
  elif -n[2] > 0.05:
    gripper_center_2 = (c - n * 0.18)
    gripper_frame[:3,2] = n
    gripper_frame[:3,3] = gripper_center_2
  else:
    return None,None

  tip_order = [0,1,2]
  if np.inner(n1,n2) > 0.8:
     gripper_frame[:3,0] = n3
     gripper_frame[:3,1] = np.cross(gripper_frame[:3,2],gripper_frame[:3,0])
     gripper_frame[:3,1] = gripper_frame[:3,1] / (np.linalg.norm(gripper_frame[:3,1]) + 1e-16)
     tip_order[2] = 3
     if np.cross((n1 - c),(n3 - c))[2] > 0:
       tip_order[0] = 1
       tip_order[1] = 2
     else:
       tip_order[0] = 2
       tip_order[1] = 1
  if np.inner(n1,n3) > 0.8:
     gripper_frame[:3,0] = n2
     gripper_frame[:3,1] = np.cross(gripper_frame[:3,2],gripper_frame[:3,0])
     gripper_frame[:3,1] = gripper_frame[:3,1] / (np.linalg.norm(gripper_frame[:3,1]) + 1e-16)
     tip_order[2] = 2
     if np.cross((n1 - c),(n2 - c))[2] > 0:
       tip_order[0] = 1
       tip_order[1] = 3
     else:
       tip_order[0] = 3
       tip_order[1] = 1

  if np.inner(n2,n3) > 0.8:
     gripper_frame[:3,0] = n1
     gripper_frame[:3,1] = np.cross(gripper_frame[:3,2],gripper_frame[:3,0])
     gripper_frame[:3,1] = gripper_frame[:3,1] / (np.linalg.norm(gripper_frame[:3,1]) + 1e-16)
     tip_order[2] = 1
     if np.cross((n2 - c),(n - c))[2] > 0:
       tip_order[0] = 2
       tip_order[1] = 3
     else:
       tip_order[0] = 3
       tip_order[1] = 2
  return gripper_frame, tip_order
 
 
if __name__ == "__main__":
  OBJ_TOP_DIR = '/home/lins/MetaGrasp/Data/Benchmarks_n/'
  DATA_TOP_DIR = '/home/lins/MetaGrasp/Data/BlensorResult/5051'
  in_mesh = [os.path.join(DATA_TOP_DIR,line) for line in os.listdir(DATA_TOP_DIR) if line.endswith('.pgm')][0]
  print(in_mesh)
  tmp = GraspConfig(graspmap_filepath=in_mesh,obj_top_dir=OBJ_TOP_DIR,friction_coef=0.5)

  grasp_config_file = [os.path.join(DATA_TOP_DIR,line) for line in os.listdir(DATA_TOP_DIR) if line.endswith('_par_3f_config.npy')][0]
  print(grasp_config_file)

  grasp_config = np.load(grasp_config_file)
  score = grasp_config[:,3]
  bestId = np.argmax(score)
  bestIdList = np.where(score > 0.0001)[0]
  contacts1 = grasp_config[:,0].astype(np.int32)
  contacts2 = grasp_config[:,1].astype(np.int32)
  contacts3 = grasp_config[:,2].astype(np.int32)

  numbest = len(bestIdList)
  print("the num of best ",numbest)
  grasp_index_file = [os.path.join(DATA_TOP_DIR,line) for line in os.listdir(DATA_TOP_DIR) if line.endswith('par_3f_512.npz')][0]
  print(grasp_index_file)

  grasp_index = np.load(grasp_index_file)

  count = 0
  for i in range(0,numbest):
    idx = bestIdList[i]
    pc = np.vstack([tmp.pcn[contacts1[idx],:3],tmp.pcn[contacts2[idx],:3],tmp.pcn[contacts3[idx],:3]])
    pcn = np.vstack([tmp.pcn[contacts1[idx],3:],tmp.pcn[contacts2[idx],3:],tmp.pcn[contacts3[idx],3:]])
    if robotiq_3f_pinch_grasp_checker(tmp.pcn[contacts1[idx],:3],tmp.pcn[contacts2[idx],:3],tmp.pcn[contacts3[idx],:3],tmp.pcn[contacts1[idx],3:],tmp.pcn[contacts2[idx],3:],tmp.pcn[contacts3[idx],3:]):
      initial_frame, tip_order = robotiq_3f_initial_frame(tmp.pcn[contacts1[idx],:3],tmp.pcn[contacts2[idx],:3],tmp.pcn[contacts3[idx],:3],tmp.pcn[contacts1[idx],3:],tmp.pcn[contacts2[idx],3:],tmp.pcn[contacts3[idx],3:])
      #  plot_pc_with_normal(pc,pcn)
      #  plot_pc(tmp.pcn[:,0:3])
      #  plot_pc(tmp.grasp_xyz,color='b')
      #  mayalab.show()
      if initial_frame is not None: 
        count = count + 1
 
  print("num cont",count)
