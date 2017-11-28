import numpy as np
import pybullet as pb
from robolearn.envs.pybullet.robot_bases import BodyPart, Joint, Camera
from robolearn.envs.pybullet.bullet_colors import pb_colors


class PyBulletRobot(object):
    def __init__(self, model_type, model_xml, base_name, action_dim, obs_dim, init_pos=None, joint_names=None, self_collision=True):
        self.model_xml = model_xml
        self.base_name = base_name
        self.self_collision = self_collision
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

    def addToScene(self, bodies):

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

        # print('The world has %d bodies.' % pb.getNumBodies())
        # print('From those, %d bodies are from the robot Mujoco file.' % len(bodies))

        if issubclass(type(bodies), int):
            bodies = [bodies]

        self._bodies_uids = bodies

        for i, bb in enumerate(bodies):
            # print('Body', bb, '--', pb.getBodyInfo(bb))

            if pb.getNumJoints(bb) == 0:
                part_name, robot_name = pb.getBodyInfo(bb, 0)
                robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(part_name, bodies, i, -1)
                # print('MJCF Body', bb, '--', 'part_name:', part_name)

            print('Bullet Robot: body id %d has %d Joints' % (bb, pb.getNumJoints(bb)))
            for j in range(pb.getNumJoints(bb)):
                # print('')
                # print('body %d=%d' % (bodies[i], bb), '| joint %d' % j)
                # pb.setJointMotorControl2(bb, j, pb.POSITION_CONTROL,
                #                          positionGain=0.1, velocityGain=0.1, force=0)
                pb.setJointMotorControl2(bb, j, controlMode=pb.POSITION_CONTROL,
                                         targetPosition=0,
                                         targetVelocity=0,
                                         positionGain=1000.1,
                                         velocityGain=10000.1,
                                         force=10000)

                joint_info = pb.getJointInfo(bb, j)
                joint_name = joint_info[1].decode("utf8")
                part_name = joint_info[12].decode("utf8")
                # print('robot_part:', part_name, '| robot_joint', joint_name)

                parts[part_name] = BodyPart(part_name, bodies, i, j)

                # Compare to base of robot
                if part_name == self.base_name:
                    self.robot_body = parts[part_name]
                    self._robot_uid = bb

                # TODO: Find a better way to define the robot_body
                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take robot_base as robot_body
                    # parts[self.robot_name] = BodyPart(self.robot_name, bodies, 0, -1)
                    # self.robot_body = parts[self.robot_name]
                    parts[self.base_name] = BodyPart(self.base_name, bodies, 0, -1)
                    self.robot_body = parts[self.base_name]
                    self._robot_uid = bb

                # If joint name starts with 'ignore' ignore it
                if joint_name[:6] == "ignore":
                    Joint(joint_name, bodies, i, j).disable_motor()
                    continue

                # If joint_name does not start with 'jointfix', then is a proper joint
                if self.joint_names is None:
                    if joint_name[:8] not in ['jointfix', 'fixed']:
                        joints[joint_name] = Joint(joint_name, bodies, i, j)
                        ordered_joints.append(joints[joint_name])

                        joints[joint_name].power_coef = 100.0
                        # joints[joint_name].power_coef = 1.0
                        print('joint', joint_name, '| lower:%f' % joints[joint_name].lowerLimit,
                              'upper:%f' % joints[joint_name].upperLimit)
                else:
                    if joint_name in self.joint_names:
                        joints[joint_name] = Joint(joint_name, bodies, i, j)
                        ordered_joints.append(joints[joint_name])

                        joints[joint_name].power_coef = 100.0
                        # joints[joint_name].power_coef = 1.0
                        print(joint_name, 'lower:%f' % joints[joint_name].lowerLimit,
                              'upper:%f' % joints[joint_name].upperLimit)
                    else:
                        temp_joint = Joint(joint_name, bodies, i, j).disable_motor()
                        print(joint_name, '(DISABLED)')

        return parts, joints, ordered_joints, self.robot_body

    def reset(self):
        # TODO: FInd a better way to recycle previous data
        self.parts, self.jdict, self.ordered_joints, self.robot_body = None, None, None, None

        # Spawn the robot again
        if self.self_collision:
            if self.model_type == 'urdf':
                model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos,
                                          flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
                # model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos,
                #                           flags=pb.URDF_USE_SELF_COLLISION+pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
            else:
                model_uid = self.load_fcn(self.model_xml,
                                          flags=pb.URDF_USE_SELF_COLLISION+pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        else:
            if self.model_type == 'urdf':
                # model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos, flags=pb.URDF_USE_INERTIA_FROM_FILE)
                model_uid = self.load_fcn(self.model_xml, basePosition=self.init_base_pos)
            else:
                model_uid = self.load_fcn(self.model_xml)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(model_uid)

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
                pb.changeVisualShape(uid, -1, rgbaColor=color_list[color_count])
                color_count += 1

            for joint in range(pb.getNumJoints(uid)):
                pb.changeVisualShape(uid, joint, rgbaColor=color_list[color_count])
                color_count += 1

    def get_total_bodies(self):
        return len(self.parts)

    def get_joint_limits(self):
        return [(joint.lowerLimit, joint.upperLimit) for joint in self.ordered_joints]

    def get_joint_torques(self):
        return [joint.get_torque() for joint in self.ordered_joints]

    def get_body_pose(self, body_name):
        return self.parts[body_name].get_pose()

    def reset_base_pos(self, position, relative_uid=-1):
        pb.resetBasePositionAndOrientation()

    def add_camera(self, body_name, dist=3, width=320, height=320):
        return Camera(self.parts[body_name], dist=dist, width=width, height=height)

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
            raise ValueError('Positions and Velocities are not the same (%d != %d)' % (len(pos), len(vel)))

        if len(pos) != len(self.ordered_joints):
            raise ValueError('State size does correspond to current robot (%d != %d)' % (len(pos)*2,
                                                                                         len(self.ordered_joints)*2))

        for jj, joint in enumerate(self.ordered_joints):
            print('Setting to joint', pos[jj], vel[jj])
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

