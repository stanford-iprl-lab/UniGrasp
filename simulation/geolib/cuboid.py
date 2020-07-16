import numpy as np



class Cuboid(object):
  def __init__(self,extrema=None,corner_points=None): 
    """ Constructor:
    @params: extrema: A numpy array with shape (6,1) [xmin,ymin,zmin,xmax,ymax,zmax]
    @params: corner_points: A numpy array with shape (8,3) 
    """
    if extrema is not None: 
      self.extrema = extrema 
       
    if corner_points is not None:
      extrema = np.zeros((6,1)) 
      extrema[0:3,0]
      extrema[0:3,0]=np.min(corner_points,axis=0)
      extrema[3:6,0]=np.max(corner_points,axis=0)
      self.extrema = extrema

    self.corner_points = self._corner_points()
 
  def _corner_points(self):
    [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
    c1 = np.array([xmin, ymin, zmin])
    c2 = np.array([xmax, ymin, zmin])
    c3 = np.array([xmax, ymax, zmin])
    c4 = np.array([xmin, ymax, zmin])
    c5 = np.array([xmin, ymin, zmax])
    c6 = np.array([xmax, ymin, zmax])
    c7 = np.array([xmax, ymax, zmax])
    c8 = np.array([xmin, ymax, zmax])
    return np.vstack([c1, c2, c3, c4, c5, c6, c7, c8])
  
  def diagonal_length(self):
    return np.linalg.norm(self.extrema[:3] - self.extrema[3:])

  def center(self):
    return np.sum(self._corner_points(), axis=0) / 8.0

  def is_point_inside(self, point):
    [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
    tmp_p = np.copy(point).reshape((3,))
    xp,yp,zp = tmp_p
    return xmin < xp and ymin < yp and zmin < zp and xmax > xp and ymax > yp and zmax > zp
    
  def are_points_inside(self, points):
    [xmin, ymin, zmin, xmax, ymax, zmax] = np.array(self.extrema).reshape((6,))
    tmp_p = np.copy(points).reshape((-1,3))
    xp = tmp_p[:,0]
    yp = tmp_p[:,1]
    zp = tmp_p[:,2]
    index = np.array([xmin < xp, ymin < yp, zmin < zp, xmax > xp, ymax > yp, zmax >zp]).transpose()
    return np.all(index,axis=1)

  def are_plane_intersect_v2(self, plane):
    plane_center = np.mean(plane,axis=0)
    cuboid_center_tmp = np.mean(self._corner_points(),axis=0)
    plane_normal = np.cross((plane[0]-plane[1]),(plane[0]-plane[2]))
    plane_normal = plane_normal/(1e-8+np.linalg.norm(plane_normal))
    cuboid_center_plane_center = cuboid_center_tmp - plane_center 
    ortho_com = np.sum(cuboid_center_plane_center * plane_normal)
    cuboid_center_2_plane_center = cuboid_center_plane_center - ortho_com * plane_normal
    cuboid_center_in_plane = plane_center + cuboid_center_2_plane_center
    tmp_corners = self._corner_points() -  cuboid_center_in_plane
    tmp_corners = tmp_corners / (1e-8 +np.linalg.norm(tmp_corners,axis=1,keepdims=True))  
    flag = np.sign(np.sum(tmp_corners * plane_normal,axis=1))   
    if np.all(flag > 0) or np.all(flag < 0):
      return False
    else:
      return True
       
  def are_plane_intersect(self, plane):
    plane_center = np.mean(plane,axis=0)
    plane_normal = np.cross((plane[0]-plane[1]),(plane[0]-plane[2]))
    tmp_corners = self._corner_points() -  plane_center
    tmp_corners = tmp_corners / (1e-8 +np.linalg.norm(tmp_corners,axis=1,keepdims=True))  
    flag = np.sign(np.sum(tmp_corners * plane_normal,axis=1))   
    if np.all(flag > 0) or np.all(flag < 0):
      return False
    else:
      return True   

  def volume(self):
    [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema.reshape((6,))
    return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)  
  
  def dim(self):
    [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema.reshape((6,))
    return [xmax-xmin,ymax-ymin,zmax-zmin] 

if __name__ == '__main__':
  c = Cuboid(extrema=[-1.0,-1.0,0.0,1.0,1.0,1.0])
  table = np.zeros((4,3))
  table[0] = np.array([-0.1,0.0,-0.01])
  table[1] = np.array([0.1,0.0,-0.01])
  table[2] = np.array([0.0,-0.1,-0.01])
  table[3] = np.array([0.0,0.1,-0.01])
  print(c.are_plane_intersect(table))
