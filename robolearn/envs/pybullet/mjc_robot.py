import numpy as np
import pybullet as pb
from pybullet_envs.robot_bases import BodyPart, Joint


class MJCFBasedRobot(object):
    def __init__(self, model_xml, base_name, action_dim, obs_dim, init_pos=None, self_collision=True):
        self.model_xml = model_xml
        self.base_name = base_name
        self.self_collision = self_collision

        if init_pos is None:
            init_pos = [0, 0, 0]
        self.init_base_pos = np.array(init_pos)

        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self._robot_uid = None

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
        for i, bb in enumerate(bodies):
            # print('Body', bb, '--', pb.getBodyInfo(bb))

            if pb.getNumJoints(bb) == 0:
                part_name, robot_name = pb.getBodyInfo(bb, 0)
                robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(part_name, bodies, i, -1)
                print('MJCF Body', bb, '--', 'part_name:', part_name)

            print('Body %d has %d Joints' % (bb, pb.getNumJoints(bb)))
            for j in range(pb.getNumJoints(bb)):
                # print('')
                # print('body %d=%d' % (bodies[i], bb), '| joint %d' % j)
                pb.setJointMotorControl2(bb, j, pb.POSITION_CONTROL,
                                         positionGain=0.1, velocityGain=0.1, force=0)
                # print(bb)
                visual_data = pb.getVisualShapeData(bb)
                print(visual_data)
                # pb.changeVisualShape(bb, j, None)

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
                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    # joints[joint_name].power_coef = 100.0
                    joints[joint_name].power_coef = 1.0
                    print(joint_name, 'lower:%f' % joints[joint_name].lowerLimit,
                          'upper:%f' % joints[joint_name].upperLimit)

        return parts, joints, ordered_joints, self.robot_body

    def reset(self):
        # TODO: FInd a better way to recycle previous data
        self.parts, self.jdict, self.ordered_joints, self.robot_body = None, None, None, None

        # Spawn the robot again
        if self.self_collision:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = \
                self.addToScene(pb.loadMJCF(self.model_xml, basePosition=self.init_base_pos,
                                            flags=pb.URDF_USE_SELF_COLLISION+pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS))
        else:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = \
                self.addToScene(pb.loadMJCF(self.model_xml, basePosition=self.init_base_pos))

        # Reset
        self.robot_reset()

        return self.robot_state()

    def robot_reset(self):
        raise NotImplementedError

    def robot_state(self):
        raise NotImplementedError


