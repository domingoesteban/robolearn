import numpy as np
from robolearn.utils.plot_utils import plot_training_costs

avg_local_policy_costs = np.load('../avg_local_policy_costsVALUES2017-07-12_07:12:22.npy')
plot_training_costs(avg_local_policy_costs, block=True)
