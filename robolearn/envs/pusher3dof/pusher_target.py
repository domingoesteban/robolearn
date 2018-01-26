import os
from robolearn.envs.pybullet.pybullet_object import PyBulletObject


class PusherTarget(PyBulletObject):
    def __init__(self, init_pos=(0., 0., 0.), type='C'):

        if type.upper() == 'C':
            urdf_file = 'target_cylinder'
        elif type.upper() == 'CS':
            urdf_file = 'target_cylinderS'
        else:
            urdf_file = 'target_sphere'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/'+urdf_file+'.urdf')

        super(PusherTarget, self).__init__('urdf', urdf_xml, 'target',
                                           init_pos=init_pos,
                                           self_collision=True)

