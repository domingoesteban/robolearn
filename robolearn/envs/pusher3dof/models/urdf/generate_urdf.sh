#!/usr/bin/env bash

xacro --inorder manipulator2d.xacro > ../pusher3dof.urdf
xacro --inorder pusher_target.xacro > ../pusher_target.urdf
xacro --inorder pusher_target2.xacro > ../pusher_target2.urdf
