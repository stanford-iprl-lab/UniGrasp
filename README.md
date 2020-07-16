# UniGrasp: Learning a Unified Model to Grasp with Multifingered Robotic Hands
To achieve a successful grasp, gripper attributes such as its geometry and kinematics play a role as important as the object
geometry. The majority of previous work has focused on developing grasp methods that generalize over novel object geometry but are
specific to a certain robot hand. We propose UniGrasp, an efficient data-driven grasp synthesis method that considers both the object
geometry and gripper attributes as inputs. UniGrasp is based on a novel deep neural network architecture that selects sets of contact
points from the input point cloud of the object. The proposed model is trained on a large dataset to produce contact points that
are in force closure and reachable by the robot hand. By using contact points as output, we can transfer between a diverse set of
multifingered robotic hands. Our model produces over 90% valid contact points in Top10 predictions in simulation and more than
90% successful grasps in real world experiments for various known two-fingered and three-fingered grippers. Our model also achieves
93%, 83% and 90% successful grasps in real world experiments for an unseen two-fingered gripper and two unseen multi-fingered
anthropomorphic robotic hands.

## Get Started
```cd models; python point_set_selection.py```


If you think our work is useful, please consider citing use with
```
@ARTICLE{8972562,
author={L. {Shao} and F. {Ferreira} and M. {Jorda} and V. {Nambiar} and J. {Luo} and E. {Solowjow} and J. A. {Ojea} and O. {Khatib} and J. {Bohg}},
journal={IEEE Robotics and Automation Letters},
title={UniGrasp: Learning a Unified Model to Grasp With Multifingered Robotic Hands},
year={2020},
volume={5},
number={2},
pages={2286-2293},
doi={10.1109/LRA.2020.2969946},
ISSN={2377-3774},
month={April},}
```
