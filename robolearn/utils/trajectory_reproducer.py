import numpy as np

class TrajectoryReproducer(object):
    def __init__(self, traj_files):

        self.trajs = []
        self.dim = 0

        for traj_file in traj_files:
            traj = np.load(traj_file)
            # Check shape
            if self.trajs:
                if traj.shape[1] != self.dim:
                    raise ValueError("Dimensions does not match! Dimension %d for file %s."
                                     "(Current dim=%d)" % (traj.shape[1], traj_file, self.dim))
                self.traj = np.vstack((self.traj, traj))
            else:
                self.traj = traj
                self.dim = traj.shape[1]

            self.trajs.append(traj.copy())


        self.data_points = self.traj.shape[0]


    def get_data(self, n):
        if n > self.data_points:
            raise ValueError("Requested point greater than available. %d > %d" % (n, self.data_points))

        return self.traj[n, :]
