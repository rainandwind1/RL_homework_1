import gym
import numpy as np
import math
import torch
from torch import nn, optim
# from DQN_py import DQN, train, plot_curse
from xzw_env import pendulum_env
# from Inverted_Pole_env import Inverted_Pole

path = "E:\Code\param\inverted_pole_testdqn.ckpt"
pi = np.math.pi
action = [-3, 0, 3]




# class Inverted_Pole():
#     def __init__(self):
#         self.Ts = 0.005
#         self.m = 0.055
#         self.g = 9.81
#         self.l = 0.042
#         self.J = 1.91*1e-4
#         self.b = 3*1e-6
#         self.K = 0.0536
#         self.R_om = 9.5
#         self.R_rew = 1.0
#         self.Q_rew = np.array([[5,0],[0,0.1]])
#         self.action = 0
        
#         self.alpha = 0.0
#         self.alpha_v = 0.0
#         self.alpha_av = 0.0
#         self.state_size = 2
#         self.action_size = 3
#         self.state = [self.alpha, self.alpha_v]

#     def init_param(self):
#         self.alpha, self.alpha_v, self.alpha_av = -pi, 0.0, 0.0
#         self.state = [self.alpha, self.alpha_v]
    
#     def is_Valid(self):
#         if self.alpha < -pi:
#             self.alpha += 2*pi
#         if self.alpha >= pi:
#             self.alpha += -2*pi
#         if self.alpha_v >= 15*pi:
#             self.alpha_v = 15*pi
#         if self.alpha_av <= -15*pi:
#             self.alpha_av = -15*pi

#     def update_param(self):
#         self.alpha_av = 1.0/self.J*(self.m*self.g*self.l*math.sin(self.alpha) - self.b*self.alpha_v - self.K**2/self.R_om*self.alpha_v + self.K/self.R_om*self.action)
#         self.is_Valid()
#         self.alpha_v += self.Ts*self.alpha_av
#         self.is_Valid()
#         self.alpha += self.Ts*self.alpha_v
#         self.is_Valid()
#         self.state = [self.alpha, self.alpha_v]

#     def reset(self):
#         self.init_param()
#         return self.state

#     def Reward(self):
#         self.state = [self.alpha, self.alpha_v]
#         reward = float(-np.mat(self.state)*self.Q_rew*np.transpose(np.mat(self.state))) - self.R_rew*self.action**2
#         return reward

#     def step(self, a):
#         self.action = action[a]
#         self.update_param()
#         reward = self.Reward()
#         s_next = self.state
#         if self.state == [0.,0.]:
#             done_flag = 0.0
#         else:
#             done_flag = 1.0
#         return s_next, reward, done_flag


# def main():
#     LOAD_KEY = False
#     learning_rate = 0.001
#     num_epochs = 3000
#     max_steps = 500
#     batch_size = 500
#     memory_len = 50000
#     replay_len = 5000
#     replay_time = 50
#     gamma = 0.98
#     train_flag = False
#     update_target_perid = 50
#     score_list = []
#     loss_list = []



#     env = pendulum_env()
#     Q_value = DQN(input_size=env.state_size, output_size=env.action_size, memory_len = memory_len)
#     Q_target = DQN(input_size=env.state_size, output_size=env.action_size, memory_len = memory_len)
#     optimizer = optim.Adam(Q_value.parameters(), lr=learning_rate)
#     loss = nn.MSELoss()

#     if LOAD_KEY:
#         checkpoint = torch.load(path)
#         Q_value.load_state_dict(checkpoint)
#         Q_target.load_state_dict(checkpoint)

#     for epo_i in range(num_epochs):
#         s = env.reset()
#         score = 0.0
#         epsilon = max(0.04,0.4-0.005*epo_i/20)
#         for i in range(max_steps):
#             a_index = Q_value.sample_action(s, epsilon)
#             s_next, reward, done_flag = env.step(a_index)
#             Q_value.save_memory((s, a_index, reward, s_next, done_flag))
#             score += reward
#             # print("action: {}, s_next: {}, reward: {}".format(a_index, s_next, reward))
#             if done_flag == 0.0:
#                 break
#         score_list.append(score)
#         if len(Q_value.memory_list) > replay_len:
#             train_flag = True
#             train(Q_value, Q_target, optimizer, loss, batch_size, gamma, loss_list, replay_time)
#         if epo_i % update_target_perid == 0 and epo_i != 0:
#             print("target net load weight!")
#             Q_target.load_state_dict(Q_value.state_dict())
#         print("{} epoch: score: {}  training: {}".format(epo_i, score, train_flag))
#     plot_curse(score_list, loss_list)
#     torch.save(Q_value.state_dict(), path)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gym

class dqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    def get_action(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done, ):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, size):
        batch = random.sample(self.memory, size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


def training(buffer, batch_size, model, optimizer, gamma, loss_fn):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = model.forward(observation)
    next_q_values = model.forward(next_observation)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0].detach()
    expected_q_value = reward + next_q_value * (1 - done) * gamma

    loss = loss_fn(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



if __name__ == '__main__':
    epsilon_init = 0.9
    epsilon_min = 0.01
    decay = 0.999
    capacity = 10000
    exploration = 500
    batch_size = 64
    episode = 1000000
    render = True
    learning_rate = 1e-3
    gamma = 0.98
    loss_fn = nn.MSELoss()
    max_step = 1200

    env = pendulum_env()
    action_dim = env.action_size
    observation_dim = env.state_size

    model = dqn(observation_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    buffer = replay_buffer(capacity)
    epsilon = epsilon_init
    weight_reward = None

    for epo_i in range(episode):
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        for i in range(max_step):
            action = model.get_action(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            train_flag = False
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if len(buffer) > exploration:
                training(buffer, batch_size, model, optimizer, gamma, loss_fn)
                train_flag = True
            if done == 0:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                break
        print('episode: {}  reward: {}   train:  {}  epsilon:{}'.format(epo_i+1, reward_total, train_flag, epsilon))
                



# if __name__ == "__main__":
#     main()


