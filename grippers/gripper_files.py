import sys
import argparse
import os
import numpy as np
import numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR,'../')
GRIPPER_DIR = os.path.join(ROOT_DIR,"grippers")
SIM_DIR = os.path.join(ROOT_DIR,"simulation")
sys.path.insert(0,SIM_DIR)

from geolib.objfile import OBJ
from geolib.cuboid import Cuboid
from urdf_parser_py.urdf import URDF
#from mayavi import mlab as mayalab
import math
import scipy.optimize

import inverse_kinematics as ik
from geolib.objfile import OBJ
from geolib.bbox import Bbox


from vis_lib import *
from grasp_reachability_nfinger import GraspConfig

_EPS = 1e-8

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def vector_norm(data, axis=None, out=None):
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)

def quaternion_about_axis(angle, axis):
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q

def quaternion_matrix(quaternion):
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def angleaxis_rotmat(angle,axis):
  quater = quaternion_about_axis(angle,axis)
  return quaternion_matrix(quater)

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


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


class Pose(object):
  def __init__(self, pose_in_urdf):
    self.xyz = pose_in_urdf.xyz
    self.rpy = pose_in_urdf.rpy
    self.T = np.zeros((4,4))
    self.T[:3,3] = self.xyz
    self.T[3,3] = 1.0
    self.T[:3,:3] = rpy_rotmat(self.rpy)


class JointLimit(object):
  def __init__(self,jointlimit_in_urdf):
    self.effort = jointlimit_in_urdf.effort
    self.velocity = jointlimit_in_urdf.velocity
    self.lower = jointlimit_in_urdf.lower
    self.upper = jointlimit_in_urdf.upper


class Mesh(object):
  def __init__(self,filename=None,scale=None):
    self.filename = filename.strip().split('file://')[1]
    self.scale = scale
    self.mesh = OBJ(file_name=self.filename)
    self.pc = self.mesh.sample_points(4096,with_normal=False)[0]

class Visual(object):
  def __init__(self,visual_in_urdf):
    self.geometry = visual_in_urdf.geometry
    self.material = visual_in_urdf.material
    if visual_in_urdf.origin != None:
      self.origin = Pose(visual_in_urdf.origin)
      self.T = self.origin.T
    else:
      self.origin = None
      self.T = np.eye(4)
    self.obj = Mesh(self.geometry.filename,scale=self.geometry.scale)
    self.obj.len_pc = len(self.obj.pc)
    self.obj.pc_4d = np.hstack([self.obj.pc,np.ones((self.obj.len_pc,1))])
    print("self.T",self.T)
    self.obj.pc_T = self.obj.pc_4d.dot(self.T.transpose())[:,0:3]


class Collision(object):
  def __init__(self,visual):
    self.epi = 0.005
    self.xmax = np.max(visual.obj.pc[:,0]) - self.epi
    self.xmin = np.min(visual.obj.pc[:,0]) + self.epi
    self.ymax = np.max(visual.obj.pc[:,1]) - self.epi
    self.ymin = np.min(visual.obj.pc[:,1]) + self.epi
    self.zmax = np.max(visual.obj.pc[:,2]) - self.epi
    self.zmin = np.min(visual.obj.pc[:,2]) + self.epi
    self.geometry = Bbox(extrema=np.array([self.xmin,self.ymin,self.zmin,self.xmax,self.ymax,self.zmax]))
    #print("self.geometry",self.geometry.corner_points)    

class Link(object):
  def __init__(self,link_in_urdf):
    self.name = link_in_urdf.name 
    self.visual = Visual(link_in_urdf.visual)
    self.collision = Collision(self.visual)
    if link_in_urdf.origin != None:
      self.origin = link_in_urdf.origin
      self.T = self.origin.T
    else:
      self.origin = None
      self.T = np.eye(4)
    self.base_T = np.eye(4)
    
  def get_transformation_matrix(self,value):
    return self.visual.T

 
class Joint(object):
  def __init__(self,joint_in_urdf):
    self.name = joint_in_urdf.name
    self.parent = joint_in_urdf.parent
    self.child = joint_in_urdf.child
    self.type = joint_in_urdf.joint_type
    self.axis = joint_in_urdf.axis
    if joint_in_urdf.origin != None:
      self.origin = Pose(joint_in_urdf.origin)
      self.T = self.origin.T
    else:
      self.origin = None
      self.T = np.eye(4)
    if joint_in_urdf.limit != None:
      self.limit = JointLimit(joint_in_urdf.limit)
    else:
      self.limit = None
    self.default_joint_value = 0
    #print("joint_name",self.name)
    #print("self.T",self.T)

  def get_transformation_matrix(self,joint_angle):
     frame = np.copy(self.T)
     print("joint_angle",joint_angle,"self.type",self.type,"self.axis",self.axis)
     if self.type == 'fixed':
       if self.axis is not None: 
         rotmat = angleaxis_rotmat(self.default_joint_value,self.axis) 
         return np.dot(frame,rotmat)
       else:
         return frame
     elif self.type == 'prismatic':
       rotmat = np.eye(4)#angleaxis_rotmat(self.default_joint_value,self.axis) 
       print("frame")
       return frame
     else:
       rotmat = angleaxis_rotmat(joint_angle,self.axis)
       frame = np.dot(frame,rotmat)
       return frame


class Chain(object):
  """
  The base Chain class
  Parameters
  ----
  links: list : List of the links for the chain
  active_links_mask:list : A list of boolean indicating that whether or not the corresponding link is active  
  name: str: The name of the Chain
  """
  def __init__(self, joints, active_joints_mask=None, name="chain", tip_pc=None, gripper=None, profile=''"",**kwargs):
    self.name = name
    self.joints = joints
    self.gripper = gripper


    # If the active_links_mask is not given, set it to True for every link
    if active_joints_mask is not None:
      self.active_joints_mask = np.array(active_joints_mask)
      # Always set the last link to True
      #self.active_joints_mask[-1] = False
    else:
      self.active_joints_mask = np.array([True] * len(joints))
    self.initial_position = np.zeros((len(self.joints),))
    self.tip_pc = tip_pc

  def forward_kinematics(self, joint_values, full_kinematics=False):
    """
    Returns the transformation matrix of the forward kinematics.
    
    Parameters:
    joints: list : The list of the positions of each joint. Noet: Inactive joints must not be in the list.
    full_kinematics: bool: Returns the transformation matrices of each joint.
    
    Returns:
    -----
    frame_matrix:
      The transformation matrix.
    """
    frame_matrix = np.eye(4)
    if self.tip_pc is not None:
      frame_matrix[:3,3] = self.tip_pc

    if full_kinematics:
      frame_matrixes = []
 
    self.joints_value = self.active_to_full(joint_values,self.initial_position)
 
    for index, joint in enumerate(self.joints):
      # Compute iteratively the position
      # NB: Use asarray to avoid old sympy problems
      if self.active_joints_mask[index]:
        frame_matrix = np.dot(np.asarray(joint.get_transformation_matrix(self.joints_value[index])),frame_matrix)
      else:
        frame_matrix = np.dot(np.asarray(joint.get_transformation_matrix(0)),frame_matrix)
             
      if full_kinematics:
        # rotation_axes = np.dot(frame_matrix, link.rotation) 
        frame_matrixes.append(frame_matrix)
  
    # Return the matrix, or matric3es
    if full_kinematics:
      return frame_matrixes
    else:
      return frame_matrix

 
  def inverse_kinematics(self,target,initial_position=None,**kwargs):
    """Computes the inverse kinematic on the speicified target.
    Parameters:
    ----------
    target: numpy.array
      The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
    initial_position: numpy.array
      Optional: the initial position of each joint of the chain. Defaults to 0 for each joint.
    
    Returns:
    ------- 
    The list of the position of each joint according to the target. Note: Inactive joints are in the list.
    """
    target = np.array(target)
    if target.shape != (4,4):
      raise ValueError("Your target must be a 4x4 transformation matrix") 
  
    if initial_position is None:
      initial_position = [0] * len(self.joints)
    else:
      if len(initial_position) != len(self.joints):
        initial_position = self.active_to_full(self.active_joints_mask,[0] * len(self.joints))

    ik_res = ik.inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)
    return ik_res

  def active_from_full(self,joints):
    return np.compress(self.active_joints_mask,joints,axis=0)

  def active_to_full(self,active_joints,intial_position):
    full_joints = np.array(intial_position,copy=True,dtype=np.float)
    np.place(full_joints,self.active_joints_mask,active_joints)
    return full_joints


class Gripper(object):
  def __init__(self,gripper_in_urdf,tip_pc):
    self.name = gripper_in_urdf.name
    self.tip_pc = tip_pc

    self.links = []
    for link in gripper_in_urdf.links:   
      self.links.append(Link(link))

    self.joints = []
    for joint in gripper_in_urdf.joints:
      self.joints.append(Joint(joint))

    self.joint_map = {}
    self.parent_map = {}
    self.child_map = {}
    self.link_map = {}

    self.q_joint_map = {}
    self.q_limit_map = {}
    self.q_T_map = {}
    self.q_value_map = {}
    self.q_axis_map = {} 
    self.q_type_map = {}

    self.chain_map = {}

    for elem in self.joints:
      self.joint_map[elem.name] = elem
      self.parent_map[elem.child] = (elem.name, elem.parent)
      if elem.parent in self.child_map:
        self.child_map[elem.parent].append((elem.name, elem.child))
      else:
        self.child_map[elem.parent] = [(elem.name, elem.child)]
 
    for elem in self.links:
      self.link_map[elem.name] = elem

    for elem in self.joints:
      if elem.type == 'revolute' or elem.type == 'prismatic':
        self.q_limit_map[elem.name] = elem.limit
        self.q_value_map[elem.name] = 0.0 #np.random.uniform(low=elem.limit.lower,high=elem.limit.upper,size=1)[0] 
        self.q_axis_map[elem.name] = elem.axis
        self.q_type_map[elem.name] = elem.type

    self.root = self.get_root()

    #for link in self.link_map:
    #  if link in self.parent_map:
    #    _, active_joints_mask, link_joint_chain = self.get_chain(self.root,link,links=False,return_active_joints=True)
    #    self.chain_map[link] = Chain(joints=link_joint_chain,name=link,active_joints_mask=active_joints_mask,tip_pc=self.tip_pc)


  def cal_configure(self):
    for elem in self.q_value_map:
      if self.q_type_map[elem] == 'revolute':
        rotmat = angleaxis_rotmat(self.q_value_map[elem],self.q_axis_map[elem])
        self.q_T_map[elem] = rotmat
      elif self.q_type_map[elem] == 'prismatic':
        self.q_T_map[elem] = angleaxis_rotmat(self.q_value_map[elem],self.q_axis_map[elem])

 
  def get_chain(self,root,tip,joints=True,links=True,fixed=True,tip_flag=True,return_active_joints=False):
    chain = []
    active_joints_mask = []
    chain_obj = []
    chain_obj.append(self.link_map[tip])
    active_joints_mask.append(False)
    if links:
      chain.append(tip)
    else:
      if tip_flag:
        chain.append(tip)
    link = tip
    while link != root:
      (joint, parent) = self.parent_map[link] 
      if joints:
        #if fixed or self.joint_map[joint].type != 'fixed':
        chain.append(joint)
        chain_obj.append(self.joint_map[joint])
        if self.joint_map[joint].type == 'fixed':
          active_joints_mask.append(False)
        else:
          active_joints_mask.append(True)
      if links:
        chain.append(parent)   
      link = parent
    if not return_active_joints:
      return chain
    else:
      return chain, active_joints_mask, chain_obj
 

  def get_root(self):
    root = None
    for link in self.link_map:
      if link not in self.parent_map:
        root = link
    return root


  def get_all_T0(self):
    self.cal_configure() 
    self.root = self.get_root()
    self.T0_map = {}
    for link in self.link_map:
      if link in self.parent_map:
        link_joint_chain = self.get_chain(self.root,link,links=False) 
        T0 = np.eye(4)
        for elem in link_joint_chain:
          if elem in self.link_map:
            T_tmp = np.copy(self.link_map[elem].visual.T)          
          if elem in self.joint_map:
            if elem in self.q_type_map:
              T_q_map = np.copy(self.q_T_map[elem])
              T_tmp = np.copy(self.joint_map[elem].T)#.dot(T_q_map)
            else:
              T_tmp = np.copy(self.joint_map[elem].get_transformation_matrix(0)) 
          T0 = T_tmp.dot(T0)
        self.T0_map[link] = self.link_map[self.root].base_T.dot(T0)
        print("link",link,self.T0_map[link])
    self.T0_map['base_link'] = self.link_map[self.root].base_T.dot(self.link_map['base_link'].visual.T)
    print("base_link",self.T0_map['base_link']) 

  def cal_all_bbox(self):
    self.cal_configure()
    self.get_all_T0()
    for link_name in self.link_map:
      link = self.link_map[link_name]
      link.collision.geometry.transformation(self.T0_map[link_name]) 
  
  def vis_all_bbox(self):
    self.cal_all_bbox()
    for link_name in self.link_map:
      link = self.link_map[link_name]  
      link.collision.geometry.plot()    
   
  def vis(self,save=False):
    self.get_all_T0()
    self.pc_whole = []
    for link_name in self.link_map:
      if link_name in self.parent_map:
        print("link_name",link_name,self.T0_map[link_name].transpose())
        link_pc = self.link_map[link_name].visual.obj.pc_4d.dot(self.T0_map[link_name].transpose())[:,0:3]
        self.pc_whole.append(link_pc)
    print(self.link_map[self.get_root()].base_T)
    self.pc_whole.append(self.link_map[self.get_root()].visual.obj.pc_4d.dot(self.link_map[self.get_root()].base_T.transpose())[:,0:3])
    self.pc_whole = np.array(self.pc_whole).reshape((-1,3))
    plot_pc(self.pc_whole)
    if save:
      np.save('robotiq3f.npy',self.pc_whole)
    #mayalab.show()


  def inverse_kinematic(self,target_tip_tf,chain_1,chain_2,chain_3,rot=False,initial_frame=None,opt_time=2,lower_stop_thres=0.018,upper_stop_thres=0.036):
     
    if not rot:     
      def optimize_target(x):
        base_frame = np.eye(4)
        base_frame[:3,:3] = rpy_rotmat(x[:3])
        base_frame[:3,-1] = x[3:6]  
        num_dof_1 = 2 
        cal_frame_1 = base_frame.dot(chain_1.forward_kinematics(x[6:6+num_dof_1]))
        target_frame_1 = target_tip_tf[0]
        dist1 = np.linalg.norm(cal_frame_1[:3,3] - target_frame_1[:3,3])  
        num_dof_2 = 2 
        cal_frame_2 = base_frame.dot(chain_2.forward_kinematics(x[6+num_dof_1:6+num_dof_1+num_dof_2]))
        target_frame_2 = target_tip_tf[1]
        dist2 = np.linalg.norm(cal_frame_2[:3,3] - target_frame_2[:3,3])  
        num_dof_3 = 1 
        cal_frame_3 = base_frame.dot(chain_3.forward_kinematics(x[6+num_dof_1+num_dof_2:6+num_dof_1+num_dof_2+num_dof_3]))
        target_frame_3 = target_tip_tf[2]
        dist3 = np.linalg.norm(cal_frame_3[:3,3] - target_frame_3[:3,3]) 
        #print("predi",cal_frame_3[:3,3],"target",target_frame_3[:3,-1]) 
        #print("dist1",dist1,"dist2",dist2,"dist3",dist3)
        return dist3 + dist2 + dist1
    else:
      def optimize_target(x):
        base_frame = np.eye(4)
        base_frame[:3,:3] = rpy_rotmat(x[:3])
        base_frame[:3,-1] = x[3:6]  
        num_dof_1 = 2 
        cal_frame_1 = base_frame.dot(chain_1.forward_kinematics(x[6:6+num_dof_1]))
        target_frame_1 = target_tip_tf[0]
        dist1 = np.linalg.norm(cal_frame_1[:3,3] - target_frame_1[:3,3])  
        dist_rot1 = np.linalg.norm(cal_frame_1[:3,1] - target_frame_1[:3,1])  
        num_dof_2 = 2 
        cal_frame_2 = base_frame.dot(chain_2.forward_kinematics(x[6+num_dof_1:6+num_dof_1+num_dof_2]))
        target_frame_2 = target_tip_tf[1]
        dist2 = np.linalg.norm(cal_frame_2[:3,3] - target_frame_2[:3,3])  
        dist_rot2 = np.linalg.norm(cal_frame_2[:3,1] - target_frame_2[:3,1])  
        num_dof_3 = 1 
        cal_frame_3 = base_frame.dot(chain_3.forward_kinematics(x[6+num_dof_1+num_dof_2:6+num_dof_1+num_dof_2+num_dof_3]))
        target_frame_3 = target_tip_tf[2]
        dist3 = np.linalg.norm(cal_frame_3[:3,3] - target_frame_3[:3,3]) 
        dist_rot3 = np.linalg.norm(cal_frame_3[:3,1] - target_frame_3[:3,1])  
        #print("dist1",dist1,"distrot1",dist_rot1,"dist2",dist2,"dist_rot2",dist_rot2,"dist3",dist3,"dist_rot3",dist_rot3)
        return dist3 + dist2 + dist1 + dist_rot1 * 0.01 + dist_rot2 * 0.01 + dist_rot3 * 0.01

    if not rot:
      def optimize_target_vis(x):
        base_frame = np.eye(4)
        base_frame[:3,:3] = rpy_rotmat(x[:3])
        base_frame[:3,-1] = x[3:6]  
        num_dof_1 = 2 
        cal_frame_1 = base_frame.dot(chain_1.forward_kinematics(x[6:6+num_dof_1]))
        target_frame_1 = target_tip_tf[0]
        dist1 = np.linalg.norm(cal_frame_1[:3,3] - target_frame_1[:3,3])  
        num_dof_2 = 2 
        cal_frame_2 = base_frame.dot(chain_2.forward_kinematics(x[6+num_dof_1:6+num_dof_1+num_dof_2]))
        target_frame_2 = target_tip_tf[1]
        dist2 = np.linalg.norm(cal_frame_2[:3,3] - target_frame_2[:3,3])  
        num_dof_3 = 1 
        cal_frame_3 = base_frame.dot(chain_3.forward_kinematics(x[6+num_dof_1+num_dof_2:6+num_dof_1+num_dof_2+num_dof_3]))
        target_frame_3 = target_tip_tf[2]
        dist3 = np.linalg.norm(cal_frame_3[:3,3] - target_frame_3[:3,3]) 
        #print("dist1",dist1,"dist2",dist2,"dist3",dist3)
        return dist3 + dist2 + dist1
    else: 
      def optimize_target_vis(x):
        base_frame = np.eye(4)
        base_frame[:3,:3] = rpy_rotmat(x[:3])
        base_frame[:3,-1] = x[3:6]  
        num_dof_1 = 2 
        cal_frame_1 = base_frame.dot(chain_1.forward_kinematics(x[6:6+num_dof_1]))
        target_frame_1 = target_tip_tf[0]
        dist1 = np.linalg.norm(cal_frame_1[:3,3] - target_frame_1[:3,3])  
        dist_rot1 = np.linalg.norm(cal_frame_1[:3,1] - target_frame_1[:3,1])  
        num_dof_2 = 2 
        cal_frame_2 = base_frame.dot(chain_2.forward_kinematics(x[6+num_dof_1:6+num_dof_1+num_dof_2]))
        target_frame_2 = target_tip_tf[1]
        dist2 = np.linalg.norm(cal_frame_2[:3,3] - target_frame_2[:3,3])  
        dist_rot2 = np.linalg.norm(cal_frame_2[:3,1] - target_frame_2[:3,1])  
        num_dof_3 = 1 
        cal_frame_3 = base_frame.dot(chain_3.forward_kinematics(x[6+num_dof_1+num_dof_2:6+num_dof_1+num_dof_2+num_dof_3]))
        target_frame_3 = target_tip_tf[2]
        dist3 = np.linalg.norm(cal_frame_3[:3,3] - target_frame_3[:3,3]) 
        dist_rot3 = np.linalg.norm(cal_frame_3[:3,1] - target_frame_3[:3,1])  
        #print("dist1",dist1,"distrot1",dist_rot1,"dist2",dist2,"dist_rot2",dist_rot2,"dist3",dist3,"dist_rot3",dist_rot3)
        return dist3 + dist2 + dist1 + dist_rot1 * 0.01 + dist_rot2 * 0.01 + dist_rot3 * 0.01
 
    
    # compute bounds
    if initial_frame is None:
      #real_bounds = [(-3.14,3.14)] * (3)
      #real_bounds += [(-0.25,0.25)] * (3)
      #real_bounds += [(0,0)] * 3 
      real_bounds = [(0,0)] * 6
    else:
      initial_rpy = euler_from_matrix(initial_frame[:3,:3])
      real_bounds = [(-3.14/10 + initial_rpy[0], 3.14/10 + initial_rpy[0])]
      real_bounds += [(-3.14/10 + initial_rpy[1], 3.14/10 + initial_rpy[1])]
      real_bounds += [(-3.14/10 + initial_rpy[2], 3.14/10 + initial_rpy[2])]
 
      real_bounds += [(-0.05 + initial_frame[0,3],0.05 + initial_frame[0,3])]
      real_bounds += [(-0.05 + initial_frame[1,3],0.05 + initial_frame[1,3])]
      real_bounds += [(-0.05 + initial_frame[2,3],0.05 + initial_frame[2,3])]
     

    for idx,joint in enumerate(chain_1.joints):
      if chain_1.active_joints_mask[idx]:
        real_bounds.append((joint.limit.lower,joint.limit.upper))
 
    for idx, joint in enumerate(chain_2.joints):
      if chain_2.active_joints_mask[idx]:
        real_bounds.append((joint.limit.lower,joint.limit.upper))
 
    for idx, joint in enumerate(chain_3.joints):
      if chain_3.active_joints_mask[idx]:
        real_bounds.append((joint.limit.lower,joint.limit.upper))

    min_fun = 1000000.0
    min_res_x = None 
   
    for i in range(opt_time):
      initial_pos = []
      for (l,h) in real_bounds:
        if h == l:
          initial_pos.append(l)
        else:
          initial_pos.append(np.random.uniform(l,h))
      #inital_pos = (6 + 4 + 4 + 3) * [0]
      res = scipy.optimize.minimize(optimize_target, initial_pos, method="L-BFGS-B",bounds=real_bounds)
      #print(res.fun)
      if res.fun < min_fun:
        min_fun = res.fun
        min_res_x = res.x
        if min_fun < lower_stop_thres or min_fun > upper_stop_thres:
          break
        #print("min_fun",min_fun,"min_x",min_res_x)
    #print("min_fun",min_fun, "min_res_x",min_res_x,"actual_",optimize_target_vis(min_res_x))
    return min_res_x, min_fun


if __name__ == "__main__":
  ### Gripper Model
  urdf_file = os.path.join(GRIPPER_DIR,"franka_hand.urdf")
  with open(urdf_file,'r') as myfile:
    urdf_strings =  myfile.read().replace('\n','')
  gripper_in_urdf = URDF.from_xml_string(urdf_strings)

  tip_pc = np.array([0.,0.,0.])
  gripper = Gripper(gripper_in_urdf=gripper_in_urdf,tip_pc=tip_pc)
  gripper.get_root()
  gripper.get_all_T0()
  gripper.vis()

  plot_origin()
  mayalab.show()
