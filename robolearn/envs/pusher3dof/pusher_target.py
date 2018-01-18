import os
from robolearn.envs.pybullet.pybullet_object import PyBulletObject


class PusherTarget(PyBulletObject):
    def __init__(self, init_pos=(0., 0., 0.)):
        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/pusher_target.urdf')

        super(PusherTarget, self).__init__('urdf', urdf_xml, 'target',
                                           init_pos=init_pos,
                                           self_collision=True)

