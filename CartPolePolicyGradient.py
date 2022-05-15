from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_score = self.fc2(x)
        return F.softmax(action_score, dim=1)   # why softmax?

class PolicyGradient(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyGradient, self).__init__()
        self.net = Net(n_states, n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-2)
        self.saved_log_prob_list = []   # 记录每个时刻的log p(a|s)
        self.reward_list = []           # 一个数组，记录每个时刻做完动作后的reward
        self.gamma = 0.99

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.net(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_prob_list.append(m.log_prob(action))  # what if continuous output?
        return action.item()

    def choose_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.net(state)
        max_action = torch.max(probs, 1)[1].data.numpy()[0]
        return max_action

    def store_transition(self, reward):
        self.reward_list.append(reward)

    def learn(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.reward_list[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_prob_list, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_prob_list.clear()
        self.reward_list.clear()

def save_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./CartPortPolicyGradient.gif', writer='imagemagick', fps=30)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # train
    done_step = 0
    learn_episode_num = 200000
    data_collect_num = 1000
    negative_reward = -10.0
    positive_reward = 10.0
    x_bound = 1.0
    state = env.reset()
    model = PolicyGradient(
        n_states=4,
        n_actions=2
    )  # 算法模型
    model.saved_log_prob_list.clear()
    model.reward_list.clear()
    running_reward = 10
    reward_threshold = 4000
    log_interval = 50
    train_mode = True

    if (train_mode):
        eps = np.finfo(np.float64).eps.item()
        for i in range(learn_episode_num):
            sum_reward = 0
            for j in range(data_collect_num):
                # env.render()
                action = model.choose_action(state)
                state, reward, done, _ = env.step(action)
                x, x_dot, theta, theta_dot = state
                if (abs(x) > x_bound):
                    r1 = 0.5 * negative_reward
                else:
                    r1 = negative_reward * abs(x) / x_bound + 0.5 * (-negative_reward)
                if (abs(theta) > env.theta_threshold_radians):
                    r2 = 0.5 * negative_reward
                else:
                    r2 = negative_reward * abs(theta) / env.theta_threshold_radians + 0.5 * (-negative_reward)
                reward = r1 + r2
                if (done) and (done_step < 499):
                    reward += negative_reward
                # print("reward: x = %lf, r1 = %lf, theta = %lf, r2 = %lf" % (x, r1, theta, r2))
                model.store_transition(reward)
                sum_reward += reward
                done_step += 1
                if (done):
                    # print("reset env! done_step = %d, epsilon = %lf" % (done_step, epsilon))
                    state = env.reset()
                    done_step = 0
                    break
            running_reward = 0.05 * sum_reward + (1 - 0.05) * running_reward
            model.learn()
            if i % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i, sum_reward, running_reward))
            if running_reward > reward_threshold:
                print("Solved! Running reward is now {} and learn_episode is {}".format(running_reward, i))
                torch.save(model, 'PolicyGradient.ptl')
                break
    else: # test mode
        frames = []
        state = env.reset()
        model_test = torch.load('PolicyGradient.ptl')
        for _ in range(400):
            frames.append(env.render(mode='rgb_array'))
            action = model_test.choose_best_action(state)
            state, reward, done, info = env.step(action)
            if (done):
                state = env.reset()
                print("test try again")
                break
        save_gif(frames)
    env.close()
