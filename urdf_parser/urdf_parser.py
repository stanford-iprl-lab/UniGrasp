
class Pose(xmlr.Object):
  def __init__(self,xyz=None,rpy=None):
    self.xyz = xyz
    self.rpy = rpy

  @property
  def rotation(self): return self.rpy
   
  @rotation.setter
  def rotation(self,value): self.rpy = value

  @property
  def position(self): return self.xyz
 
  @position.setter
  def position(self,value): self.xyz = value


class JointDynamics(xmlr.Object):
  def __init__(self,damping=None,friction=None):
    self.damping = damping
    self.friction = friction

class Cylinder(xmlr.Object):
  def __init__(self, radius=0.0, length=0.0):
    self.radius = radius
    self.length = length

class Box(xmlr.Object):
  def __init__(self,size=None):
    self.size = size

class Sphere(xmlr.Object):
  def __init__(self, radius=0.0):
    self.radius = radius

class Mesh(xmlr.Object):
  def __init__(self,filename=None,scale=None):
    self.filename = filename
    self.scale = scale


class  GeometricType(xmlr.ValueType):
  def __init__(self):
    self.factory = xmlr.FactoryType('geometry',{
       'box':Box,
       'cylinder': Cylinder,
       'sphere': Sphere,
       'mesh': Mesh
    })

  def from_xml(self,node):
    children = xml_children(node)
    assert len(children) == 1, 'One element only for geometric'
    return self.factory.from_xml(children[0])

 def write_xml(self,node,obj):
   name = self.factory.get_name(obj)
   child = node_add(node, name)
   obj.write_xml(child)

class Collision(xmlr.Object):
  def __init__(self,geometry=None,origin=None):
    self.geometry = geometry
    self.origin = origin


class Visual(xmlr.Object):
  def __init__(self,geometry=None,material=None,origin=None):
    self.geometry = geometry
    self.material = material 
    self.origin = origin


class Inertia(xmlr.Object):
  KEYS = ['ixx','iyy','izz','ixy','iyz','ixz']
 
  def __init__(self,ixx=0.0,ixy=0.0,ixz=0.0,iyy=0.0,iyz=0.0,izz=0.0):
    self.ixx = ixx
    self.ixy = ixy
    self.ixz = ixz
    self.iyy = iyy
    self.iyz = iyz
    self.izz = izz

  def to_matrix(self):
    return [ 
       [self.ixx, self.ixy, self.ixz],
       [self.ixy, self.iyy, self.iyz],
       [self.ixz, self.iyz, self.izz]]



class Inertial(xmlr.Object):
  def __init__(self,mass=0.0,inertia=None,origin=None):
    self.mass = mass
    self.inertia = inertia
    self.origin = origin


class JointCalibration(xmlr.Object):
  def __init__(self, rising=None, falling=None):
    self.rising = rising
    self.falling = falling


class JointLimit(xmlr.Object):
  def __init__(self,eff)

class SafetyController(xmlr.Object):
  def __init__(self,velocity=None,position=None,lower=None,upper=None):
    self.k_velocity = velocity
    self.k_position = position
    self.soft_lower_limit = lower
    self.soft_upper_limit = upper


class Joint(xmlr.Object):
  TYPES = ['unknown','revolute','continuous','prismatic','floating','planar','fixed']

  def __init__(self, name=None, parent=None, child=None, joint_type=None, axis=None, origin=None,
        limit=None,dynamics=None,safety_controller=None,calibration=None,mimic=None):
    self.name = name
    self.parent = parent
    self.child = child
    self.type = joint_type
    self.axis = axis
    self.origin = origin
    self.limit = limit
    self.dynamics = dynamics
    self.safety_controller = safety_controller
    self.calibration = calibration
    self.mimic = mimic
  
  def child_valid(self):
    assert self.type in self.TYPES, "Invalid joint type: {}".format(self.type)

 

class Link(xmlr.Object):
  def __init__(self,name=None,visual=None,inertial=None,colllision=None,origin=None):
    self.name = name  
    self.visual = visual
    self.inertial = inertial
    self.colllision = collision
    self.origin = origin


class Robot(xmlr.Object):
  def __init__(self,name=None):
    self.aggregate_init()

    self.name = name
    self.joints = []
    self.links = []
    self.materials = []
    self.gazebos = []
    self.transmission = []

    self.joint_map = {}
    self.link_map = {}
   
    self.parent_map = {}
    self.child_map = {}

  def add_aggregate(self, typeName, elem):
    xmlr.Object.add_aggregate(self,typeName,elem)

    if typeName == 'joint':
      joint = elem
      self.joint_map[joint.name] = joint
      self.parent_map[joint.child] = (joint.name, joint.parent)
      if joint.parent in self.child_map:
        self.child_map[joint.parent].append((joint.name,joint.child))
      else:
        self.child_map[joint.parent] = [(joint.name, joint.child)]
    elif typeName == 'link':
      link = elem
      self.link_map[link.name] = link


   def add_link(self,link):
     self.add_aggregate('link',link)

   def add_joint(self,joint):
     self.add_aggregate('joint',joint)

   def get_chain(self, root, tip, joints=True, links=True, fixed=True):
     chain = []
     if links:
       chain.append(tip)
     link = tip
     while link != root:
       (joint, parent) = self.parent_map[link]
       if joints:
         if fixed or self.joint_map[joint].joint_type != 'fixed':
           chain.append(joint)
         if links:
           chain.append(parent)
         link = parent
      chain.reverse()
      return chain

    def get_root(self):
      root = None
      for link in self.link_map:
        if link not in self.parent_map:
          assert root is None, "Multiple roots detected, invalid URDF."
          root = link
      assert root is not None, "No roots detected, invalid URDF"
      return root
