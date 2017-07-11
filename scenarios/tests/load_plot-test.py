import numpy as np
import matplotlib.pyplot as plt

avg_local_policy_costs = np.load('../avg_local_policy_costsVALUES2017-07-10_23:32:53.npy')
t = np.arange(avg_local_policy_costs.shape[0])
plt.plot(t, np.average(avg_local_policy_costs, axis=1))
plt.fill_between(t, np.min(avg_local_policy_costs, axis=1), np.max(avg_local_policy_costs, axis=1), alpha=0.5)
plt.show(block=True)
