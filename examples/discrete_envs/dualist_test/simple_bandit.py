import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)
np.random.seed(0)


k = 4  # N bandits
time_steps = 10  # T
max_iters = 2

actions = np.arange(k)
rewards = np.array([10, -2, -10, 2])
avg_rewards = np.zeros_like(actions)
action_counter = np.zeros_like(actions)
sum_rewards = np.zeros_like(actions)

temp1 = 5
temp2 = 5

w1 = 0.5
w2 = 0.5

min_reward = np.inf
max_reward = -np.inf

expected = rewards.mean()
q_vals1 = np.zeros_like(actions, dtype=np.float64)
q_vals2 = np.zeros_like(actions, dtype=np.float64)
q_vals = np.zeros_like(actions, dtype=np.float64)
p1 = np.ones_like(actions)/len(actions)
p2 = np.ones_like(actions)/len(actions)
policy = (w1*p1 + w2*p2)/(w1 + w2)

width = 0.25
fig = plt.figure(1)
ax1 = fig.add_subplot(3, 1, 1)
bar1 = ax1.bar(actions, p1, color='b')
text1 = [ax1.text(nn, p1[nn] + .05, str(p1[nn]), color='b',
                  horizontalalignment='center') for nn in range(len(policy))]
ax1.set_title('Initial policy')
ax1.set_ylim([0, 1])
ax1.set_xticks(actions)
ax1.set_xticklabels(actions.astype(str))

ax2 = fig.add_subplot(3, 1, 2)
bar2 = ax2.bar(actions, p2, color='r')
# ax2.set_title('Policy 2')
text2 = [ax2.text(nn, p2[nn] + .05, str(p2[nn]), color='r',
                  horizontalalignment='center') for nn in range(len(policy))]
ax2.set_ylim([0, 1])
ax2.set_xticks(actions)
ax2.set_xticklabels(actions.astype(str))

ax = fig.add_subplot(3, 1, 3)
bar = ax.bar(actions, policy, color='k')
# ax2.set_title('Policy 2')
text = [ax.text(nn, policy[nn] + .05, str(policy[nn]), color='k',
                horizontalalignment='center') for nn in range(len(policy))]
ax.set_ylim([0, 1])
ax.set_xticks(actions)
ax.set_xticklabels(actions.astype(str))

# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(1, 1, 1)
# bar2 = ax2.bar(actions, p2, color='r')
# ax2.set_title('Policy 2')
# ax2.set_ylim([0,1])

plt.subplots_adjust(wspace=0, hspace=0)
plt.show(block=False)

user_input = 'whatever'

for itr in range(max_iters):
    fig.canvas.set_window_title("'%d'-Bandit problem | itr:%02d" % (k, itr))

    user_input = input("Start itr %02d:" % itr)
    ax1.set_title('Policy at itr = %02d' % itr)

    for t in range(time_steps):
        ax1.set_title('t = %02d' % t)


        # ACTION SELECTION

        # Energy-based policy
        # temp = 0.1*t + 1.e-1
        # temp1 = temp2 = temp

        p1 = np.exp(q_vals1/temp1)/np.exp(q_vals1/temp1).sum()
        p2 = np.exp(q_vals2/temp2)/np.exp(q_vals2/temp2).sum()

        policy = (w1*p1 + w2*p2)/(w1 + w2)
        action = np.random.choice(actions, p=policy)

        # Bandit reward
        # reward = rewards[action]
        reward = np.random.normal(rewards[action], 0.5)

        # ax1.set_title('Action: %d | reward: %.2f' % (action, reward))

        # POLICY EVALUATION
        user_input = input("t=%d | a=%d | Q_vals=%s || Press 'f' to finish:" %
                           (t, action, q_vals))
        if user_input.lower() in ['f']:
            break

        # Sample-average Q estimation
        # ix = np.where(action_counter > 0)
        # q_vals[ix] = sum_rewards[ix]/action_counter[ix]  # Non-efficient way
        # action_counter[action] += 1

        # Incremental formula
        action_counter[action] += 1
        q_error = reward - q_vals[action]
        alpha = 1./action_counter[action]  # Step size
        q_vals[action] = q_vals[action] + alpha * q_error

        # q_vals = rewards

        q_vals1 = q_vals2 = q_vals

        sum_rewards[action] += reward
        t += 1



        # Visualize
        for nn in range(len(policy)):
            bar1[nn].set_height(p1[nn])
            bar2[nn].set_height(p2[nn])
            bar[nn].set_height(policy[nn])
            text1[nn].set_position((nn, p1[nn]+0.05))
            text2[nn].set_position((nn, p2[nn]+0.05))
            text[nn].set_position((nn, policy[nn]+0.05))
            text1[nn].set_text(str(np.around(p1[nn], 2)))
            text2[nn].set_text(str(np.around(p2[nn], 2)))
            text[nn].set_text(str(np.around(policy[nn], 2)))

        # print(action)
        fig.canvas.draw_idle()


print("Script has finished")
