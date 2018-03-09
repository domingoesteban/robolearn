import numpy as np
import pybullet as pb
import logging
from robolearn.envs.pybullet.robot_bases import BodyPart, Joint, Camera
from robolearn.envs.pybullet.bullet_colors import pb_colors


class PyBulletRobot(object):
    def __init__(self, model_type, model_xml, base_name, action_dim, obs_dim,
                 init_pos=None, joint_names=None, self_collision=True,
                 use_file_inertia=True):
        self.model_xml = model_xml
        self.base_name = base_name
        self.self_collision = self_collision
        self.use_file_intertia = use_file_inertia
        self.model_type = model_type
        self.joint_names = joint_names

        if self.model_type == 'urdf':
            self.load_fcn = pb.loadURDF
        elif self.model_type == 'mjcf':
            self.load_fcn = pb.loadMJCF
        elif self.model_type == 'sdf':
            self.load_fcn = pb.loadSDF
        else:
            raise NotImplemented('Wrong model_type. Only URDF and MJCF are supported')

        if init_pos is None:
            init_pos = [0, 0, 0]
        self.init_base_pos = np.array(init_pos)

        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self._robot_uid = None

        self._bodies_uids = None


        # high = np.ones([action_dim])
        # self.action_space = gym.spaces.Box(-high, high)
        # high = np.inf * np.ones([obs_dim])
        # self.observation_space = gym.spaces.Box(-high, high)
        self.action_dim = action_dim
        self.observation_dim = obs_dim

        # Logger
        self.logger = logging.getLogger('pybullet')
        self.logger.setLevel(logging.WARNING)

    def addToScene(self, bodies):
        self.logger.info('*'*40)
        self.logger.info('pbROBOT | Adding robot to scene')

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        self.logger.info('pbROBOT | At this moment, the world has %d bodies.'
                         % pb.getNumBodies())
        # print('From those, %d bodies are from the robot Mujoco file.' % len(bodies))

        if not isinstance(bodies, tuple):
            bodies = [bodies]

        self._bodies_uids = bodies

        self.logger.info('pbROBOT | Evaluating the %d bodies of file.')
        for i, bb in enumerate(bodies):
            # print('Body', bb, '--', pb.getBodyInfo(bb))
            self.logger.info('pbROBOT | body id %d has %d Joints'
                             % (bb, pb.getNumJoints(bb)))

            if pb.getNumJoints(bb) == 0:
                part_name, robot_name = pb.getBodyInfo(bb, 0)
                robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(part_name, bodies, i, -1)
                self.logger.info('pbROBOT | Body id: %d has 0 joints and '
                                 'part_name %s has been added to parts dict.'
                                 % (bb, part_name))
                if len(bodies) == 1:
                    self.logger.warning('pbROBOT | This is not a robotBody but '
                                        'creating robot_body and robot_uid '
                                        'anyway.')
                    self.robot_body = parts[part_name]
                    self._robot_uid = bb

            for j in range(pb.getNumJoints(bb)):
                self.logger.info('pbROBOT | Joint %d' % j)
                # print('body %d=%d' % (bodies[i], bb), '| joint %d' % j)
                # pb.setJointMotorControl2(bb, j, pb.POSITION_CONTROL,
                #                          positionGain=0.1,
                #                          velocityGain=0.1,
                #                          force=0)
                pb.setJointMotorControl2(bb, j, controlMode=pb.POSITION_CONTROL,
                                         targetPosition=0,
                                         targetVelocity=0,
                                         positionGain=1000.1,
                                         velocityGain=10000.1,
                                         force=10000)

                joint_info = pb.getJointInfo(bb, j)
                joint_name = joint_info[1].decode("utf8")
                part_name = joint_info[12].decode("utf8")

                self.logger.info('robot_joint:%s moves part_name:%s'
                                 % (joint_name, part_name))

                parts[part_name] = BodyPart(part_name, bodies, i, j)
                self.logger.info('Adding part_name %s to parts dict'
                                 % part_name)

                # Compare to base of robot
                if self.robot_body is None:
                    if part_name == self.base_name:
                        self.logger.info('part_name matches base_name! '
                                         'Using it as robot_body and uid')
                        self.robot_body = parts[part_name]
                        self._robot_uid = bb
                    elif j == 0:  # i == 0 and j == 0:
                        self.logger.warning('First joint (%s) does not match '
                                            'base_name (%s)! Using it as '
                                            'robot_body and uid'
                                            % (part_name, self.base_name))
                        parts[self.base_name] = BodyPart(self.base_name,
                                                         bodies, 0, -1)
                        self.robot_body = parts[self.base_name]
                        self._robot_uid = bb
                    else:
                        self.logger.error('Problem defining robot_body')
                        raise AttributeError('Error defining robot_body')
                    self.logger.info('The robot_body is now: %s'
                                     % self.robot_body.body_name)

                # If joint name starts with 'ignore' ignore it
                if joint_name[:6] == "ignore":
                    Joint(joint_name, bodies, i, j).disable_motor()
                    continue

                if self.joint_names is None:
                    # Defining active joints using name
                    if joint_name[:8] not in ['jointfix', 'fixed']:
                        joints[joint_name] = Joint(joint_name, bodies, i, j)
                        ordered_joints.append(joints[joint_name])

                        joints[joint_name].power_coef = 100.0
                        # joints[joint_name].power_coef = 1.0
                        self.logger.info('joint %s | lower:%f upper%f'
                                         % (joint_name,
                                         joints[joint_name].lowerLimit,
                                         joints[joint_name].upperLimit))
                else:
                    # Defining active joints with self.joint_names
                    if joint_name in self.joint_names:
                        joints[joint_name] = Joint(joint_name, bodies, i, j)
                        ordered_joints.append(joints[joint_name])

                        joints[joint_name].power_coef = 100.0
                        # joints[joint_name].power_coef = 1.0
                        self.logger.info('joint %s | lower:%f upper%f'
                                         % (joint_name,
                                            joints[joint_name].lowerLimit,
                                            joints[joint_name].upperLimit))
                    else:
                        temp_joint = Joint(joint_name, bodies, i, j).disable_motor()
                        self.logger.info('joint %s DISABLED.' % joint_name)

        return parts, joints, ordered_joints, self.robot_body

    def reset(self):
        # TODO: FInd a better way to recycle previous data
        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            None, None, None, None

        # TODO: Use self.use_file_inertia for urdf

        # Spawn the robot again
        if self.self_collision:
            if self.model_type == 'urdf':
                model_uid = \
                    self.load_fcn(self.model_xml, basePosition=self.init_base_pos,
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
                model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos)
            else:
                model_uid = self.load_fcn(self.model_xml)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            self.addToScene(model_uid)

        # print('')
        # print('parts:', self.parts)
        # print('ordered_j:', self.ordered_joints)
        # print('jdict:', self.jdict)


        # Reset
        self.robot_reset()

        return self.robot_state()

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

    def get_total_joints(self):
        return len(self.ordered_joints)

    def get_joint_limits(self):
        return [(joint.lowerLimit, joint.upperLimit) for joint in self.ordered_joints]

    def get_joint_torques(self):
        return [joint.get_torque() for joint in self.ordered_joints]

    def get_body_pose(self, body_name):
        return self.parts[body_name].get_pose()

    def reset_base_pos(self, position, relative_uid=-1):
        pb.resetBasePositionAndOrientation()

    def add_camera(self, body_name, dist=3, width=320, height=320):
        return Camera(self.parts[body_name], dist=dist,
                      width=width, height=height)

    def robot_reset(self):
        raise NotImplementedError

    def robot_state(self):
        raise NotImplementedError

    def set_robot_pose(self, position, orientation=(0, 0, 0, 1)):
        self.robot_body.reset_pose(position, orientation)

    def get_robot_pose(self):
        return self.robot_body.get_pose()

    def set_state(self, pos, vel):
        if len(pos) != len(vel):
            raise ValueError('Positions and Velocities are not the same '
                             '(%d != %d)' % (len(pos), len(vel)))

        if len(pos) != len(self.ordered_joints):
            raise ValueError('State size does correspond to current robot '
                             '(%d != %d)' % (len(pos)*2,
                                             len(self.ordered_joints)*2))

        for jj, joint in enumerate(self.ordered_joints):
            joint.set_state(pos[jj], vel[jj])

    def get_state(self):
        njoints = len(self.ordered_joints)
        state = np.zeros(njoints*2)
        for jj, joint in enumerate(self.ordered_joints):
            state[jj], state[jj+njoints] = joint.get_state()

        return state

    def set_color(self, color):
        if issubclass(type(color), str):
            if color.lower() in pb_colors:
                color = pb_colors[color]
        color_list = [color for _ in range(self.get_total_bodies())]
        self.set_body_colors(color_list)

    def get_jacobian(self, link_name, joint_pos, joint_vel=None,
                     joint_acc=None, link_point=None):

        jacobian = np.zeros((6, len(self.ordered_joints)))

        if link_name not in self.parts:
            raise ValueError("Link name %s is not a robot body_part"
                             % link_name)

        if link_point is None:
            link_point = [0., 0., 0.]

        pos = [0]*pb.getNumJoints(self._robot_uid)
        vel = [0]*pb.getNumJoints(self._robot_uid)
        torq = [0]*pb.getNumJoints(self._robot_uid)

        if joint_vel is None:
            joint_vel = np.zeros_like(joint_pos)

        if joint_acc is None:
            joint_acc = np.zeros_like(joint_pos)

        for jj in range(len(self.ordered_joints)):
            pb_joint_id = self.ordered_joints[jj].jointIndex
            pos[pb_joint_id] = joint_pos[jj]
            vel[pb_joint_id] = joint_vel[jj]
            torq[pb_joint_id] = joint_acc[jj]

        jac_t, jac_r = pb.calculateJacobian(self._robot_uid,
                                   self.parts['gripper_center'].bodyPartIndex,
                                   link_point, pos, vel, torq)

        for dd, (tt, rr) in enumerate(zip(jac_t, jac_r)):
            jacobian[dd, :] = tt
            jacobian[dd+3, :] = rr

        return jacobian
