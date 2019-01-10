import os

import argparse
import joblib
import uuid
from robolearn.utils.logging import logger
import json
import numpy as np
import robolearn.torch.utils.pytorch_util as ptu
import torch

filename = str(uuid.uuid4())
SEED = 110


def convert_to_cpu(args):

    np.random.seed(SEED)
    ptu.seed(SEED)

    for file in os.listdir(args.dir):
        if file.endswith(".pkl"):
            if file == "params.pkl" and not args.params:
                continue
            full_file = os.path.join(args.dir, file)
            data = joblib.load(full_file)

            if args.gpu >= 0:
                device_name = "cuda:%f" % int(args.gpu)
            else:
                device_name = "cpu"

            print("Converting to %s: %s" % (device_name, full_file))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    if args.gpu >= 0:
                        data[key] = value.cuda(int(args.gpu))
                    else:
                        data[key] = value.cpu()

            joblib.dump(data, full_file, compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='path to the snapshot file')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU id')
    parser.add_argument('--params', action="store_true")
    args = parser.parse_args()

    convert_to_cpu(args)
    print('The script has finished')
