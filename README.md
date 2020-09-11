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
1.Initialize repository
```
git clone https://github.com/stanford-iprl-lab/UniGrasp.git
```

2.Compile pointnet++
```
cd models/poinetnet4
```

The TF operators are included under tf_ops, you need to compile them (run tf_xxx_compile.sh under each ops subfolder).

3.Run
```
cd models; python point_set_selection.py
```

## Grasp Objects With The Pretrained Model.
1. If you have your own gripper, you will need to generate the point clouds of the gripper under specific joint configurations described in the paper. 
Here is a github repository I used in UniGrasp. https://github.com/linsats/Python-Parser-for-Robotic-Gripper

2. Generate the gripper features after feeding these point clouds in the gripper auto-encoder model using gripper_feature_extraction.py 

3. Send the object point cloud and the gripper features into the point set selection network.The pretrained model is in [point_set_selection.zip](http://download.cs.stanford.edu/juno/UniGrasp/pretrained_models/point_set_selection.zip). Download and unzip it under the UniGrasp/saved_models. Then run the code point_set_selection_test.py

4. Solve in the inverse kinematic given the position and orientation of the selected contact points. We use the RBDL, https://github.com/rbdl/rbdl.


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
