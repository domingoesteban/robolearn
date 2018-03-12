import numpy as np
from robolearn.utils.experience_buffer import ExperienceBuffer

N_trajs = 8
T = 2
buffer_size = 3
good_or_bad = 'good'
temp_or_cost = 'cost'

all_trajs = list()
all_costs = list()
for ii in range(10):
    all_trajs.append(np.ones(T)*ii)
    all_costs.append(all_trajs[-1]*10)

index_order = np.arange(N_trajs)
np.random.shuffle(index_order)

experience_buffer = ExperienceBuffer(buffer_size, good_or_bad, temp_or_cost)

for ii in index_order:
    print(all_trajs[ii])

print('##' * 20)

for nn in range(len(index_order) // 2):
    trajs_to_add = [all_trajs[index_order[2 * nn]],
                    all_trajs[index_order[2 * nn + 1]]]
    costs_to_add = [all_costs[index_order[2 * nn]],
                    all_costs[index_order[2 * nn + 1]]]
    print('trajs', trajs_to_add)
    print('costs', costs_to_add)
    experience_buffer.add(trajs_to_add, costs_to_add)
    print('len', len(experience_buffer))
    print('btrajs', experience_buffer._trajs)
    print('bcosts', experience_buffer._costs)
    print('--')


print('TRAJS WITH OPS', experience_buffer.get_trajs(2))
print('COsts WITH OPS', experience_buffer.get_costs(2))
print('BOTH WITH OPS', experience_buffer.get_trajs_and_costs(2))

