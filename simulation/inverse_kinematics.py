import scipy.optimize
import numpy as np

def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None,max_iter=None,rot=False):
  """Compute the inverse kinematic on the specified target with an optimization method.
   Parameters:
   --- 
   chain: ikpy.chain.Chain: The chain used for the Inverse kinematics.
   target: numpy.array : The desired target.
   starting_nodes_angles: numpy.array : The initial pose of your chain.
   regualarization_parameter: float : The coefficient of the regularization.
   max_iter: int:  Maximum number of iterations for the optimization algorithm..
   """
  # Only get the position
  target = target_frame[:3,3]
  # Compute squared distance to target:
  if not rot:
    def optimize_target(x):
      #y = chain.active_to_full(x, starting_nodes_angles)
      squared_distance = np.linalg.norm(chain.forward_kinematics(x)[:3,3] - target)
      return squared_distance
  else:
    def optimize_target(x):
      cal_frame = chain.forward_kinematics(x)
      squared_distance = np.linalg.norm(cal_frame[:3,3] - target_frame[:3,3])
      squared_rot = np.linalg.norm(cal_frame[:3,:3] - target_frame[:3,3])
      return squared_distance + squared_rot

  # If a regularization is selected
  if regularization_parameter is not None:
    def optimize_total(x):
      regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])  
      return optimize_target(x) + regularization_parameter * regularization
  else:
    def optimize_total(x):
      return optimize_target(x)
  
  # Compute bounds
  real_bounds = []
  real_bounds.append((0,0))
  for joint in chain.joints[1:]:
    if joint.limit is not None:
      real_bounds.append((joint.limit.lower,joint.limit.upper))
    else:
      real_bounds.append((0,0))
  
  print(real_bounds) 
  # real_bounds 
  real_bounds = chain.active_from_full(real_bounds)

  options = {}
  # Manage iterations maximum
  if max_iter is not None:
    options["maxiter"] = max_iter

  # Utilisation 
  res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles),method="L-BFGS-B",bounds=real_bounds) 
  return res.x
