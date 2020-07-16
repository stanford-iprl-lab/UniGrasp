import numpy as np


def obj_loader(file_name,normalize=False):
  """It loads the vertices, vertice normals, the faces of a wavefront obj file.
  """
  vertices = []
  faces = []
  vnormals = []

  with open(file_name,'r') as fin:
    for line in fin:
      if line.startswith('#'):
        continue
      values = line.split()
      if len(values) < 1:
        continue
      if values[0] == 'v':
        v = map(float,values[1:4])
        vertices.append(v)
      elif values[0] == 'vn':
        vn = map(float,values[1:4])
        vnormals.append(vn)
      elif values[0] == 'f':
        face = []
        for v in values[1:]:
          w = v.split('/')
          face.append(int(w[0])) 
        faces.append(face) 
  
  vertices = np.array(vertices)
  faces = np.array(faces)
  vnormals = np.array(vnormals)
  faces = faces-1
  if normalize:
    bbox_max = np.max(vertices,axis=0)
    bbox_min = np.min(vertices,axis=0)
    bbox_center = 0.5 * (bbox_max + bbox_min)
    bbox_rad =  np.linalg.norm(bbox_max - bbox_center)
    vertices -= bbox_center
    vertices /= (bbox_rad*2.0)
  if np.any(faces < 0):     
    print('Negative face indexing in obj file')
  
  return vertices, faces, vnormals

if __name__ == "__main__":
  v,f,vn = obj_loader('/Users/lins/GraspNet/mesh_processing/reorient_faces_coherently/vase.obj',normalize=True)
