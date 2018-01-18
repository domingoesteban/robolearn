import numpy as np
import pybullet as pb
import logging
from robolearn.envs.pybullet.robot_bases import BodyPart, Joint, Camera
from robolearn.envs.pybullet.bullet_colors import pb_colors


class PyBulletObject(object):
    def __init__(self, model_type, model_xml, base_name,
                 init_pos=None, self_collision=True, use_file_inertia=True):
        self.model_xml = model_xml
        self.base_name = base_name
        self.self_collision = self_collision
        self.use_file_intertia = use_file_inertia
        self.model_type = model_type

        if self.model_type == 'urdf':
            self.load_fcn = pb.loadURDF
        elif self.model_type == 'mjcf':
            self.load_fcn = pb.loadMJCF
        elif self.model_type == 'sdf':
            self.load_fcn = pb.loadSDF
        else:
            raise NotImplemented('Wrong model_type.'
                                 'Only URDF and MJCF are supported')

        if init_pos is None:
            init_pos = [0, 0, 0]
        self.init_base_pos = np.array(init_pos)

        self.parts = None
        self.object_body = None
        self._object_uid = None

        self._bodies_uids = None

        # Logger
        self.logger = logging.getLogger('pybullet_object')
        self.logger.setLevel(logging.WARNING)

    def addToScene(self, bodies):
        self.logger.info('*'*40)
        self.logger.info('pbOBJECT | Adding object to scene')

        self.parts = {}

        self.logger.info('pbOBJECT | At this moment, the world has %d bodies.'
                         % pb.getNumBodies())

        if issubclass(type(bodies), int):
            bodies = [bodies]

        self._bodies_uids = bodies

        self.logger.info('pbOBJECT | Evaluating the %d bodies of file.')
        for i, bb in enumerate(bodies):
            # print('Body', bb, '--', pb.getBodyInfo(bb))
            self.logger.info('pbOBJECT | body id %d has %d Joints'
                             % (bb, pb.getNumJoints(bb)))

            if pb.getNumJoints(bb) == 0:
                part_name, object_name = pb.getBodyInfo(bb, 0)
                object_name = object_name.decode("utf8")
                part_name = part_name.decode("utf8")
                self.parts[part_name] = BodyPart(part_name, bodies, i, -1)
                self.logger.info('pbOBJECT | Body id: %d has 0 joints and '
                                 'part_name %s has been added to parts dict.'
                                 % (bb, part_name))
                if len(bodies) == 1:
                    self.logger.warning('pbOBJECT | This is not a robotBody '
                                        'but creating object_body and object_uid '
                                        'anyway.')
                    self.object_body = self.parts[part_name]
                    self._object_uid = bb

    def reset(self):
        # TODO: FInd a better way to recycle previous data
        self.parts, self.object_body = None, None

        # TODO: Use self.use_file_inertia for urdf

        # Spawn the robot again
        if self.self_collision:
            if self.model_type == 'urdf':
                model_uid = \
                    self.load_fcn(self.model_xml,
                                  basePosition=self.init_base_pos,
                                  flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                                        +pb.URDF_USE_INERTIA_FROM_FILE)
                # model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos,
                #                           flags=pb.URDF_USE_SELF_COLLISION+pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
            else:
                model_uid = \
                    self.load_fcn(self.model_xml,
                                  flags=pb.URDF_USE_SELF_COLLISION
                                        +pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
                                        +pb.URDF_USE_INERTIA_FROM_FILE)
        else:
            if self.model_type == 'urdf':
                # model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos, flags=pb.URDF_USE_INERTIA_FROM_FILE)
                model_uid = self.load_fcn(self.model_xml,
                                          basePosition=self.init_base_pos)
            else:
                model_uid = self.load_fcn(self.model_xml)

        self.addToScene(model_uid)

        return self.get_pose()

    def set_body_colors(self, color_list):
        color_count = 0
        for uid in self._bodies_uids:
            if pb.getNumJoints(uid) == 0:
                pb.changeVisualShape(uid, -1,
                                     rgbaColor=color_list[color_count])
                color_count += 1

            for joint in range(pb.getNumJoints(uid)):
                pb.changeVisualShape(uid, joint,
                                     rgbaColor=color_list[color_count])
                color_count += 1

    def get_total_bodies(self):
        return len(self.parts)

    def get_body_pose(self, body_name):
        return self.parts[body_name].get_pose()

    def reset_base_pos(self, position, relative_uid=-1):
        pb.resetBasePositionAndOrientation()

    def add_camera(self, body_name, dist=3, width=320, height=320):
        return Camera(self.parts[body_name], dist=dist,
                      width=width, height=height)

    def set_pose(self, position, orientation=(0, 0, 0, 1)):
        self.object_body.reset_pose(position, orientation)

    def get_pose(self):
        return self.object_body.get_pose()

    def set_color(self, color):
        if issubclass(type(color), str):
            if color.lower() in pb_colors:
                color = pb_colors[color]
        color_list = [color for _ in range(self.get_total_bodies())]
        self.set_body_colors(color_list)
