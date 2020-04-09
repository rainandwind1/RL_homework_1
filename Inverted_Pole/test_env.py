import gym
import numpy as np
import math
import torch
from torch import nn, optim
from DQN_py import DQN, train, plot_curse
# from Inverted_Pole_env import Inverted_Pole

pi = np.math.pi
action = [-3, 0, 3]

class Inverted_Pole():
    def __init__(self):
        self.Ts = 0.005
        self.m = 0.055
        self.g = 9.81
        self.l = 0.042
        self.J = 1.91*1e-4
        self.b = 3*1e-6
        self.K = 0.0536
        self.R_om = 9.5
        self.R_rew = 1.0
        self.Q_rew = np.array([[5,0],[0,0.1]])
        self.action = 0
        
        self.alpha = 0.0
        self.alpha_v = 0.0
        self.alpha_av = 0.0
        self.state_size = 2
        self.action_size = 3
        self.state = [self.alpha, self.alpha_v]

    def init_param(self):
        self.alpha, self.alpha_v, self.alpha_av = -pi, 0.0, 0.0
        self.state = [self.alpha, self.alpha_v]
    
    def is_Valid(self):
        if self.alpha < -pi:
            self.alpha += 2*pi
        if self.alpha >= pi:
            self.alpha += -2*pi
        if self.alpha_v >= 15*pi:
            self.alpha_v = 15*pi
        if self.alpha_av <= -15*pi:
            self.alpha_av = -15*pi

    def update_param(self):
        self.alpha_av = 1.0/self.J*(self.m*self.g*self.l*math.sin(self.alpha) - self.b*self.alpha_v - self.K**2/self.R_om*self.alpha_v + self.K/self.R_om*self.action)
        self.is_Valid()
        self.alpha_v += self.Ts*self.alpha_av
        self.is_Valid()
        self.alpha += self.Ts*self.alpha_v
        self.is_Valid()
        self.state = [self.alpha, self.alpha_v]

    def reset(self):
        self.init_param()
        return self.state

    def Reward(self):
        self.state = [self.alpha, self.alpha_v]
        reward = float(-np.mat(self.state)*self.Q_rew*np.transpose(np.mat(self.state))) - self.R_rew*self.action**2
        return reward

    def step(self, a):
        self.action = action[a]
        self.update_param()
        reward = self.Reward()
        s_next = self.state
        if self.state == [0.,0.]:
            done_flag = 0.0
        else:
            done_flag = 1.0
        return s_next, reward, done_flag


def main():
    learning_rate = 0.0001
    num_epochs = 600
    max_steps = 500
    batch_size = 500
    memory_len = 50000
    replay_len = 5000
    replay_time = 50
    gamma = 0.98
    train_flag = False
    update_target_perid = 50
    score_list = []
    loss_list = []



    env = Inverted_Pole()
    Q_value = DQN(input_size=env.state_size, output_size=env.action_size, memory_len = memory_len)
    Q_target = DQN(input_size=env.state_size, output_size=env.action_size, memory_len = memory_len)
    optimizer = optim.Adam(Q_value.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    for epo_i in range(num_epochs):
        s = env.reset()
        score = 0.0
        epsilon = max(0.01,0.2-0.01*epo_i/200)
        for i in range(max_steps):
            a_index = Q_value.sample_action(s, epsilon)
            s_next, reward, done_flag = env.step(a_index)
            Q_value.save_memory((s, a_index, reward, s_next, done_flag))
            score += reward
            # print("action: {}, s_next: {}, reward: {}".format(a_index, s_next, reward))
            if done_flag == 0.0:
                break
        score_list.append(score)
        if len(Q_value.memory_list) > replay_len:
            train_flag = True
            train(Q_value, Q_target, optimizer, loss, batch_size, gamma, loss_list, replay_time)
        if epo_i % update_target_perid == 0 and epo_i != 0:
            print("target net load weight!")
            Q_target.load_state_dict(Q_value.state_dict())
        print("{} epoch: score: {}  training: {}".format(epo_i, score, train_flag))
    plot_curse(score_list, loss_list)


if __name__ == "__main__":
    main()


