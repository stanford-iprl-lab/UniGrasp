import warnings
import numpy as np
from objloader import obj_loader
from math import sin, cos

def plot_pc(pcs,color=None,scale_factor=.05,mode='point'):
  if color == 'red':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(1,0,0))
  elif color == 'blue':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,0,1))
  elif color == 'green':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,0))
  elif color == 'ycan':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,1))
  else:
    print("unkown color")
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=color)

def plot_pc_with_normal(pcs,pcs_n,scale_factor=1.0):
  mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], mode='arrow',scale_factor=1.0)


def plot_origin():
  origin_pc = np.array([0.0,0.0,0.0]).reshape((-1,3))
  plot_pc(origin_pc,color='ycan',mode='sphere',scale_factor=.01)
  origin_pcs = np.tile(origin_pc,(3,1))
  origin_pcns = np.eye(3) * 0.01
  plot_pc_with_normal(origin_pcs,origin_pcns)


try:
  from mayavi import mlab as mayalab 
except:
  warnings.warn('Mayavi library was not found.')

class OBJ(object):
  def __init__(self, vertices=None, faces=None, vnormals=None, normal_normalized=True,normalize=False,file_name=None,scale=1.0,rotation_degree=0):
    self.vertices = None
    self.faces = None 
    self.vnormals = None
    self.fnormals = None
    self.fareas = None
    self.seed = 42
    self.normalize = normalize 

    if file_name is not None:
      self.vertices, self.faces, self.vnormals = obj_loader(file_name,self.normalize)[:3]      
    else:
      self.vertices = vertices
      self.faces = faces 
      self.vnormals = vnormals
 
    if abs(scale - 1.0) > 0.0000001: 
      self.vertices *= scale
 
    if abs(rotation_degree - 0.0) > 0.0001:
      new_x = self.vertices[:,0]*cos(rotation_degree / 180.0 * np.pi) + self.vertices[:,1]*sin(rotation_degree / 180.0 * np.pi)
      new_y = self.vertices[:,0]*sin(rotation_degree / 180.0 * np.pi) - self.vertices[:,1]*cos(rotation_degree / 180.0 * np.pi)
      self.vertices = np.vstack([new_x,new_y*-1.0,self.vertices[:,2]]).T
 
    self.num_points = len(self.vertices)
    self.num_faces = len(self.faces)
     
    if normal_normalized:
      self.cal_vnormals()
  
  def centers(self):
    return np.mean(self.vertices,axis=0)
   
  def cal_fareas(self):
    self.fareas = 0.5 * np.linalg.norm( np.cross(self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,1],:], self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,2],:]),axis=1)
    return np.copy(self.fareas) 

  def cal_fnormals(self): 
    N = np.cross(self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,1],:], self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,2],:])
    row_norms = np.linalg.norm(N,axis=1)
    self.fareas = 0.5 * row_norms
    N = (N.T / row_norms).T
    self.fnormals = N

  def cal_vnormals(self):
    if self.vnormals.shape != self.vertices.shape:
      N = np.cross(self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,1],:], self.vertices[self.faces[:,0],:] - self.vertices[self.faces[:,2],:]) 
      self.vnormals = np.empty_like(self.vertices)
      self.vnormals[self.faces[:,0],:] = N 
      self.vnormals[self.faces[:,1],:] += N
      self.vnormals[self.faces[:,2],:] += N
    row_norms = np.linalg.norm(self.vnormals,axis=1) 
    self.vnormals = (self.vnormals.T / row_norms).T 
     
  def write(self,file_name):
    self.faces = self.faces + 1
    with open(file_name,'w') as fw:
      for v in self.vertices:
        fw.write('v %f %f %f\n' % (v[0],v[1],v[2]))
      
      if self.vnormals.shape == self.vertices.shape:
        for vn in self.vnormals:
          fw.write('vn %f %f %f\n' % (vn[0],vn[1],vn[2])) 
   
      for f in self.faces:
        fw.write('f %d %d %d\n' % (f[0],f[1],f[2]))

  def sample_points(self,n_samples,seed=None,with_normal=False):
    if seed is not None:   
      np.random.seed(seed)
    else:
      np.random.seed(self.seed)

    face_areas = self.cal_fareas()
    face_areas = face_areas / np.sum(face_areas)
 
    n_samples_per_face = np.round(n_samples * face_areas)
 
    n_samples_per_face = n_samples_per_face.astype(np.int)
    n_samples_s = int(np.sum(n_samples_per_face))

    diff = n_samples_s - n_samples
    indices = np.arange(self.num_faces)
 
    if diff > 0:
      rand_faces = np.random.choice(indices[n_samples_per_face >= 1], abs(diff), replace=False)
      n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] - 1
    elif diff < 0:
      rand_faces = np.random.choice(indices, abs(diff), replace=False)
      n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] + 1

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
     
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
      sample_face_idx[acc:acc + _n_sample] = face_idx
      acc += _n_sample
    
    r = np.random.rand(n_samples, 2)
    
    A = self.vertices[self.faces[sample_face_idx, 0], :]
    B = self.vertices[self.faces[sample_face_idx, 1], :]
    C = self.vertices[self.faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
          np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    if with_normal:
      self.cal_vnormals()
      An = self.vnormals[self.faces[sample_face_idx,0],:]
      Bn = self.vnormals[self.faces[sample_face_idx,1],:]
      Cn = self.vnormals[self.faces[sample_face_idx,2],:]

      An[np.isnan(An)] = 0.0
      Bn[np.isnan(Bn)] = 0.0
      Cn[np.isnan(Cn)] = 0.0

      Pn = (1 - np.sqrt(r[:, 0:1])) * An + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * Bn + \
            np.sqrt(r[:, 0:1]) * r[:, 1:] * Cn
      row_norms = np.linalg.norm(Pn, axis=1)
      row_norms[row_norms == 0] = 1
      Pn = (Pn.T / row_norms).T
      return P, Pn, sample_face_idx
    else:
      return P, sample_face_idx
  
  def plot_normals(self):
    mayalab.quiver3d(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.vnormals[:, 0], self.vnormals[:, 1], self.vnormals[:, 2], mode='arrow')

  def plot_sample_points(self):
    self.s_pc,self.s_pc_n = self.sample_points(1000,with_normal=True)[0:2]
    mayalab.quiver3d(self.s_pc[:, 0], self.s_pc[:, 1], self.s_pc[:, 2], self.s_pc_n[:, 0], self.s_pc_n[:, 1], self.s_pc_n[:, 2], mode='arrow')
 
  def plot(self,triangle_function=np.array([]),vertex_function=np.array([]),show=True,*args,**kwargs):
    mayalab.triangular_mesh(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.faces, color=(209/255.0,64/255.0,109/255.0), *args, **kwargs)
   
if __name__ == "__main__":
  #test = OBJ(file_name='/home/lins/MetaGrasp/Data/Benchmarks_n/003.obj',scale=0.0001,rotation_degree=-45.0,normalize=False)
  #test = OBJ(file_name="/home/lins/bullet3/examples/pybullet/gym/pybullet_data/Panda/robotiq_2f_85/visual/robotiq_gripper_coupling.obj",rotation_degree=0.0,normalize=False,scale=0.001)
  test = OBJ(file_name="/home/lins/Downloads/model.obj",rotation_degree=0.0,normalize=False)
  test.vertices -= test.centers()
  #test.cal_vnormals()
  #tmp = test.sample_points(10000,with_normal=False)[0]
  #print(tmp.shape)
  #plot_pc(tmp,color=(209/255.0,64/255.0,109/255.0),mode='sphere',scale_factor=0.005)
  #tmp = np.copy(test.vnormals[:,2])
  #test.vnormals[:,2] = np.copy(test.vnormals[:,1])
  #test.vnormals[:,1] = np.copy(tmp)
  #test.vnormals *= -1.0

  #test.vertices[:,2] *= -1.0
  #tmp = np.copy(test.vertices[:,2])
  #test.vertices[:,2] = np.copy(test.vertices[:,1])
  #test.vertices[:,1] = np.copy(tmp) 
 
  #test.vertices[:,0] -= 0.04 
  #test.vertices[:,1] -= 0.04

  #print(test.vnormals.shape)
  #test.sample_points(1000,with_normal=True)
  #test.plot()
  #test.plot_normals()
  #test.plot_sample_points()
  #plot_origin()
  #mayalab.show()
  test.write('Shovel.obj')
