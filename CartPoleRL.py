import gym
from matplotlib import animation
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
        out = self.fc2(x)
        return out

class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions) # nit two nets
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_actions = n_actions
        self.n_states = n_states
        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((2000, 2 * n_states + 1 + 1))  # s, s', a, r
        self.cost = []  # 记录损失值
        self.done_step_list = []

    def choose_action(self, state, epsilon):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)
        if np.random.uniform() < epsilon:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0] # d the max value in softmax layer. before .data, it is a tensor
        else:
            action = np.random.randint(0, self.n_actions)
        # print("action=", action)
        return action

    def store_transition(self, state, action, reward, next_state):
        # print("<store_transition>")
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % 2000  # 满了就覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
            # print("update eval to target")
        self.learn_step_counter += 1

        # 使用记忆库中批量数据
        sample_index = np.random.choice(2000, 16)  # 200个中随机抽取32个作为batch_size
        memory = self.memory[sample_index, :]  # 取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :self.n_states])
        action = torch.LongTensor(memory[:, self.n_states:self.n_states + 1])
        reward = torch.LongTensor(memory[:, self.n_states + 1:self.n_states + 2])
        next_state = torch.FloatTensor(memory[:, self.n_states + 2:])

        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action) # eval_net->(64,4)->按照action索引提取出q_value
        q_next = self.target_net(next_state).detach()
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + 0.5 * q_next.max(1)[0].unsqueeze(1) # label
        loss = self.loss(q_eval, q_target)  # td error
        self.cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

    def plot_cost(self):
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")

        plt.subplot(1,2,2)
        plt.plot(np.arange(len(self.done_step_list)), self.done_step_list)
        plt.xlabel("step")
        plt.ylabel("done step")
        plt.show()

def save_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./CartPortCtrl.gif', writer='imagemagick', fps=30)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    frames = []
    # train
    counter = 0
    done_step = 0
    max_done_step = 0
    num = 200000
    rec_num = 10
    negative_reward = -10.0
    x_bound = 1.0
    max_done_step_result = []
    done_step_list_result = []
    for retime in range(rec_num):
        state = env.reset()
        model = DQN(
            n_states=4,
            n_actions=2
        )  # 算法模型
        model.cost.clear()
        model.done_step_list.clear()
        for i in range(num):
            # env.render()
            # frames.append(env.render(mode='rgb_array'))
            epsilon = 0.9 + i / num * (0.95 - 0.9)
            # epsilon = 0.9
            action = model.choose_action(state, epsilon)
            # print('action = %d' % action)
            state_old = state
            state, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = state
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # x_threshold 4.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            if (abs(x) > x_bound):
                r1 = 0.5 * negative_reward
            else:
                r1 = negative_reward * abs(x) / x_bound + 0.5 * (-negative_reward)
            if (abs(theta) > env.theta_threshold_radians):
                r2 = 0.5 * negative_reward
            else:
                r2 = negative_reward * abs(theta) / env.theta_threshold_radians + 0.5 * (-negative_reward)
            reward = r1 + r2
            if done:
                reward += negative_reward
            # print("x = %lf, r1 = %lf, theta = %lf, r2 = %lf" % (x, r1, theta, r2))
            model.store_transition(state_old, action, reward, state)
            if (i > 2000 and counter % 10 == 0):
                model.learn()
                counter = 0
            counter += 1
            done_step += 1
            if (done):
                # print("reset env! done_step = %d, epsilon = %lf" % (done_step, epsilon))
                if (done_step > max_done_step):
                    max_done_step = done_step
                state = env.reset()
                model.done_step_list.append(done_step)
                done_step = 0
        #model.plot_cost()  # 误差曲线
        print("reccurent time = %d, max done step = %d, final done step = %d" % (retime, max_done_step, model.done_step_list[-1]))
        max_done_step_result.append(max_done_step)
        done_step_list_result.append(model.done_step_list[-1])
    # test
    '''
    state = env.reset()
    for _ in range(2000):
        frames.append(env.render(mode='rgb_array'))
        action = model.choose_action(state, 1.0)
        state, reward, done, info = env.step(action)
        if (done):
            state = env.reset()
            print("test try again")
    env.close()
    '''
    # save_gif(frames)




