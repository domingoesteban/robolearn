#!/usr/bin/env bash

xacro --inorder manipulator2d.xacro > ../pusher3dof.urdf
xacro --inorder target_cylinder.xacro > ../target_cylinder.urdf
xacro --inorder target_cylinderS.xacro > ../target_cylinderS.urdf
xacro --inorder target_sphere.xacro > ../target_sphere.urdf
