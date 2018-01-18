#!/usr/bin/env bash

xacro --inorder manipulator2d.xacro > ../pusher3dof.urdf
xacro --inorder pusher_target.xacro > ../pusher_target.urdf
