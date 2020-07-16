from mayavi import mlab as mayalab 
import numpy as np
import os

def plot_pc(pcs,color=None,scale_factor=.05,mode='point'):
  if color == 'red':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(1,0,0))
    print("color",color)
  elif color == 'blue':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,0,1))
  elif color == 'green':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,0))
  elif color == 'ycan':
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=(0,1,1))
  else:
    print("unkown color")
    mayalab.points3d(pcs[:,0],pcs[:,1],pcs[:,2],mode=mode,scale_factor=scale_factor,color=color)

  
def plot_pc_with_normal(pcs,pcs_n,scale_factor=1.0,color='red'):
   if color == 'red':
     mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], color=(1,0,0), mode='arrow',scale_factor=1.0)
   elif color == 'blue':
     mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], color=(0,0,1), mode='arrow',scale_factor=1.0)
   elif color == 'green':
     mayalab.quiver3d(pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs_n[:, 0], pcs_n[:, 1], pcs_n[:, 2], color=(0,1,0), mode='arrow',scale_factor=1.0)


def plot_origin():
  origin_pc = np.array([0.0,0.0,0.0]).reshape((-1,3))
  plot_pc(origin_pc,color='ycan',mode='sphere',scale_factor=.01)
  origin_pcs = np.tile(origin_pc,(3,1))
  origin_pcns = np.eye(3) * 0.01
  plot_pc_with_normal(origin_pcs,origin_pcns)


if __name__ == '__main__':
 #save_dir = '/home/lins/MetaGrasp/Data/BlensorResult/2056'
 #gripper_name = '056_rho0.384015_azi1.000000_ele89.505854_theta0.092894_xcam0.000000_ycam0.000000_zcam0.384015_scale0.146439_xdim0.084960_ydim0.084567_zdim0.08411000000_pcn_new.npz.npy'

 #gripper_name ='339_rho0.308024_azi6.000000_ele89.850030_theta-0.013403_xcam0.000000_ycam0.000000_zcam0.308024_scale0.061975_xdim0.048725_ydim0.036192_zdim0.01252500000_pcn.npz'
 gripper = np.load(os.path.join("robotiq2f_open.npy"))
 #plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.002)
 plot_pc(gripper,color=(209/255.0,64/255.0,109/255.0),mode='sphere',scale_factor=0.002)
 plot_origin()
 mayalab.show()
 #sle = np.array([1494,1806])
 #plot_pc(gripper[sle],color='red',mode='sphere',scale_factor=0.002)
 #mayalab.show()
 
 #save_dir = '/home/lins/MetaGrasp/meta_grasping/saved_results/interp'
 #save_dir = '/home/lins/MetaGrasp/Data/Gripper/Data3'
 # #save_dir_gt = '/home/lins/MetaGrasp/Data/Gripper/Data'

 save_dir = '/home/lins/MetaGrasp/Data/Gripper/Data_DB/G5/f2_5_close.npy'
 a = np.load(save_dir)
 plot_pc(a)
 save_dirb = '/home/lins/MetaGrasp/Data/Gripper/Data_DB/G3/f2_3_close.npy'
 b = np.load(save_dirb)
 plot_pc(b,color='red')
 mayalab.show()
 #for i in range(10001,10300):
 # gripper_name = 'f2_'+str(i)+'_middel.npy'
  #print(gripper_name)
 # gripper = np.load(os.path.join(save_dir,gripper_name))
 # plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.002)
 # plot_origin()
 # mayalab.show()
  #save_dir_gt = '/home/lins/MetaGrasp/Data/Gripper/Data'
  #gripper_gt = np.load(os.path.join(save_dir_gt,gripper_name))
  #plot_pc(gripper_gt,color='red',mode='sphere',scale_factor=0.002)
 
 if 0:
  for i in range(0,199):
   save_dir = '/home/lins/MetaGrasp/Data/Gripper/Data_noR'
 #save_dir = '/home/lins/MetaGrasp/meta_grasping/saved_results/recon_old'
   gripper_name = 'robotiq_3f_'+str(i)+'.npy'
   print(gripper_name)
   gripper = np.load(os.path.join(save_dir,gripper_name))
   plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)

   plot_origin()
   mayalab.show()
 
 
 if 0:
  save_dir = '/home/lins/MetaGrasp/meta_grasping/saved_results/interp'
  gripper_name = 'kinova_kg3_0.npy'
  print(gripper_name)
  gripper = np.load(os.path.join(save_dir,gripper_name))
  plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)

  plot_origin()
  mayalab.show()
  
  gripper_name = 'robotiq_3f_1.npy'
  print(gripper_name)
  gripper = np.load(os.path.join(save_dir,gripper_name))
  plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)
  plot_origin()
  mayalab.show()

  save_dir = '/home/lins/MetaGrasp/meta_grasping/saved_results/interp'
  gripper_name = 'middle0.npy'
  print(gripper_name)
  gripper = np.load(os.path.join(save_dir,gripper_name))
  plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)

  plot_origin()
  mayalab.show()
  
  gripper_name = 'middle1.npy'
  print(gripper_name)
  gripper = np.load(os.path.join(save_dir,gripper_name))
  plot_pc(gripper,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)
  plot_origin()
  mayalab.show()
 
  save_dir = '/home/lins/MetaGrasp/Data/Gripper/Data_noR'
  gripper_name1 = 'kinova_kg3_0.npy'
  print(gripper_name)
  gripper1 = np.load(os.path.join(save_dir,gripper_name1))
  plot_pc(gripper1,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)
  plot_origin()
  mayalab.show()
 
  gripper_name2 = 'robotiq_3f_1.npy'
  print(gripper_name)
  gripper2 = np.load(os.path.join(save_dir,gripper_name2))
  plot_pc(gripper2,color=(139/255.0,177/255.0,212/255.0),mode='sphere',scale_factor=0.01)

  plot_origin()
  mayalab.show()
