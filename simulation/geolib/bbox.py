import numpy as np
from mayavi import mlab as mayalab

def plot_pc_with_normal(pcs,pcs_n,scale_factor=1.0):
  mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], mode='arrow',scale_factor=1.0)

def plot_pc(pcs,color=None,scale_factor=.05,mode='point'):
  if color == 'r':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(1,0,0))
  elif color == 'blue':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,0,1))
  elif color == 'green':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,0))
  elif color == 'ycan':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,1))
  else:
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor)


class Bbox(object):
  def __init__(self,extrema=None,corner_points=None,frame_length=None):
    """
    z   -x
    |   /
    |  /
    | / 
    0-----> y

      c5
      | --/--> c8 
      |  /
      | / 
   c6   c1-----> c4
   |  /
   | /
   |/ 
   c2 -----> c3

   """
    if extrema is not None:
      [xmin, ymin, zmin, xmax, ymax, zmax] = extrema
      self.c1 = np.array([xmin, ymin, zmin])
      self.c2 = np.array([xmax, ymin, zmin])
      self.c3 = np.array([xmax, ymax, zmin])
      self.c4 = np.array([xmin, ymax, zmin])
      self.c5 = np.array([xmin, ymin, zmax])
      self.c6 = np.array([xmax, ymin, zmax])
      self.c7 = np.array([xmax, ymax, zmax])
      self.c8 = np.array([xmin, ymax, zmax])
      self.corner_points = np.vstack([self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8])
      self.frame_length = np.array([xmax-xmin,ymax-ymin,zmax-zmin])
    else:
      self.corner_points = corner_points
      self.frame_length = frame_length 
    self.frame_rot = np.eye(4)
    self.transformation(np.eye(4))
   
  def transformation(self,T):
    self.corner_points_4d = np.hstack([np.copy(self.corner_points),np.ones((8,1))])
    self.corner_points_transformed = self.corner_points_4d.dot(T.transpose())[:,0:3]
  
  def is_point_inside(self, point, transformed=True):
    if transformed:
      self.basis = np.zeros((3,3)) 
      self.basis[:,0] = self.corner_points_transformed[1]-self.corner_points_transformed[0]
      self.basis[:,1] = self.corner_points_transformed[3]-self.corner_points_transformed[0]
      self.basis[:,2] = self.corner_points_transformed[4]-self.corner_points_transformed[0]
      self.basis[:,0] = self.basis[:,0] / (np.linalg.norm(self.basis[:,0]) + 1e-16)
      self.basis[:,1] = self.basis[:,1] / (np.linalg.norm(self.basis[:,1]) + 1e-16)
      self.basis[:,2] = self.basis[:,2] / (np.linalg.norm(self.basis[:,2]) + 1e-16)

      point = point - self.corner_points_transformed[0]
      point = point.dot(self.basis)
      if point[0] < self.frame_length[0] and point[1] < self.frame_length[1] and  point[2] < self.frame_length[2] and point[0] > 0 and point[1] > 0  and point[2] > 0:
        return True
      else:
        return False

  def points_inside(self, points_o, transformed=True):
    if transformed:
      self.basis = np.zeros((3,3)) 
      self.basis[:,0] = self.corner_points_transformed[1]-self.corner_points_transformed[0]
      self.basis[:,1] = self.corner_points_transformed[3]-self.corner_points_transformed[0]
      self.basis[:,2] = self.corner_points_transformed[4]-self.corner_points_transformed[0]
      self.basis[:,0] = self.basis[:,0] / (np.linalg.norm(self.basis[:,0]) + 1e-16)
      self.basis[:,1] = self.basis[:,1] / (np.linalg.norm(self.basis[:,1]) + 1e-16)
      self.basis[:,2] = self.basis[:,2] / (np.linalg.norm(self.basis[:,2]) + 1e-16)

      transformed_o = np.copy(self.corner_points_transformed[0]).reshape((1,3))
       
      points = points_o - transformed_o
      points = points.dot(self.basis)
      points_check = np.hstack([points[:,0:1] < self.frame_length[0], points[:,1:2] < self.frame_length[1] , points[:,2:3] < self.frame_length[2] ,  points[:,0:1] > 0 ,points[:,1:2] > 0, points[:,2:3] > 0])
      points_flag = np.all(points_check,axis=1)
      if np.any(points_flag):
        #pct = np.vstack([transformed_o,transformed_o,transformed_o])
        #plot_pc_with_normal(pct,self.basis.transpose() * 0.01)
        #plot_pc(self.corner_points_transformed,color='ycan',scale_factor=0.01,mode='sphere')
        #plot_pc(points_o[points_flag],color='r',scale_factor=0.01,mode='sphere')
        #print(points_check[points_flag])
        #print(points[points_flag])
        return True
      else:
        return False
 
  def rect_intersect(self,table=None):
    if np.any(self.corner_points_transformed[:,2] < 0.0):
      return True
    else:
      return False
    
  def plot(self,transformed=True):
    if transformed:
      p1 = self.corner_points_transformed[0]
      p2 = self.corner_points_transformed[1]
      p3 = self.corner_points_transformed[2]
      p4 = self.corner_points_transformed[3]

      p5 = self.corner_points_transformed[0]
      p6 = self.corner_points_transformed[1]
      p7 = self.corner_points_transformed[2]
      p8 = self.corner_points_transformed[3]

      p9 = self.corner_points_transformed[4]
      p10 = self.corner_points_transformed[5]
      p11 = self.corner_points_transformed[6]
      p12 = self.corner_points_transformed[7]

      c1 = p1 - p2
      c2 = p2 - p3
      c3 = p3 - p4
      c4 = p4 - p1

      c5 = p1 - p9
      c6 = p2 - p10
      c7 = p3 - p11
      c8 = p4 - p12

      c9 = p9 - p10
      c10 = p10 - p11
      c11 = p11 - p12
      c12 = p12 - p9

      ps = np.vstack([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
      cs = np.vstack([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12])
      cs = cs * -1.0 
      mayalab.quiver3d(ps[:,0],ps[:,1],ps[:,2],cs[:,0],cs[:,1],cs[:,2],mode='2ddash',scale_factor=1.0)      

 

if __name__ == '__main__':
  tmp = Bbox(extrema=np.array([-1.0,-1.0,-1.0,1.0,1.0,1.0])*0.1) 
  tmp.transformation(np.eye(4))
  print(tmp.is_point_inside(np.array([0.0,0,0])))
  #tmp.plot()
  #mayalab.show()
