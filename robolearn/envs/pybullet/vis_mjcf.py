import os
import inspect

import pybullet as p
import pybullet_data
import time


def test(args):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if args.mjc_model is None:
        fileName = os.path.join('mjcf', args.mjcf)
    else:
        fileName = currentdir + '/mjc_models/' + args.mjc_model

    print('*'*10)
    print("fileName: %s" % fileName)
    print('*'*10)

    p.loadMJCF(fileName)
    while True:
        p.stepSimulation()
        p.getCameraImage(320, 240)
        time.sleep(0.01)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mjcf', help='MJCF filename (mjcf dir)', default="humanoid.xml")
    parser.add_argument('--mjc_model', help='MJCF filename (mjc_models dir)', default=None)
    args = parser.parse_args()
    test(args)
