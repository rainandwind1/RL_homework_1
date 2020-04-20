import turtle as t
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xzw_env import pendulum_env # 训练用的环境
import copy
pi = np.math.pi

def Data_is_vaild(alpha, alpha_v):
    if alpha < -pi:
        alpha += 2*pi
    if alpha >= pi:
        alpha += -2*pi
    if alpha_v <= -15*pi:
        alpha_v = -15*pi
    if alpha_v >= 15*pi:
        alpha_v = 14.99*pi
    return alpha, alpha_v


# 随机动作采样数据以备学习
sample_size = 15000
Q_rew = np.matrix([[5.,0.],[0.,0.1]])
R_rew = 1.0
a = [-3,0,3]
T_s = 0.005
# al_range = [-pi,pi)
# al_v_range = [-15*pi,15*pi]
def random_play():
    env = pendulum_env()
    mem = []
    alpha_bat = np.random.uniform(-pi, pi, sample_size)
    alpha_v_bat = np.random.uniform(-15*pi, 15*pi, sample_size)
    a_index = np.random.choice(env.action_size, sample_size)
    for alpha, alpha_v, a_idx in zip(alpha_bat, alpha_v_bat, a_index):
        state = np.mat([alpha, alpha_v])
        alpha_av = (1.0/env.j)*(env.m*env.g*env.l*math.sin(alpha) - env.b*alpha_v - env.k**2/env.r*alpha_v + env.k/env.r*a[a_idx])
        alpha_v_next = alpha_v + T_s*alpha_av
        alpha_next = alpha + T_s*alpha_v
        alpha_next, alpha_v_next = Data_is_vaild(alpha_next, alpha_v_next)
        state_next = [alpha_next, alpha_v_next]
        reward = float(-np.matrix(state_next)*Q_rew*np.transpose(np.matrix(state_next))) - R_rew*a[a_idx]**2
        mem.append([[alpha, alpha_v], a_idx, reward, state_next])
    return mem

# test_data = random_play()
# print(test_data)

def Gauss_RBF(u1, u2, sigma1, sigma2, s):
    return math.exp(-0.5*(np.mat(s)-np.mat([u1,u2]))*np.mat([[sigma1**2,0.],[0.,sigma2**2]]).I*(np.mat(s)-np.mat([u1,u2])).T)


sigma1 = pi/4
sigma2 = 3.75*pi
def x_state(s, a):
    cha = []
    for m in range(discrete_num):
        for n in range(discrete_num):
            u1 = -pi + (m+1)*2*pi/(discrete_num+1)
            u2 = -15*pi + (n+1)*30*pi/(discrete_num+1)
            cha.append(Gauss_RBF(u1, u2, sigma1, sigma2, s))
    if a == 0:
        return np.mat(cha + [0. for i in range(discrete_num**2)]*2).T
    elif a == 1:
        return np.mat([0. for i in range(discrete_num**2)] + cha + [0. for i in range(discrete_num**2)]).T
    else:
        return np.mat([0. for i in range(discrete_num**2)]*2 + cha).T


discrete_num = 9
w_init = [0.0 for i in range(discrete_num*discrete_num*3)]
train_steps = 10
gamma = 0.98
flag_step = 0
LOAD_KEY = False
def train():
    w = np.mat(w_init).T
    if LOAD_KEY:
        w = np.load('lspi.npy')
    data = random_play()
    X_s = []
    X_next = []
    R = []
    for n_index,trans in enumerate(data):
        s, a_idx, r, s_next = trans
        next_val = []
        R.append(r)
        for i in range(3):
            next_val.append(x_state(s_next, i))
        X_s.append(x_state(s, a_idx))
        X_next.append(next_val)

    print("train beginning!")
    policy = [0 for i in range(sample_size)]
    policy_pre = [1 for i in range(sample_size)]
    iter_count = 0
    while iter_count<200:
        iter_count += 1
        w_pre = w
        count = 0
        sigma_Y_t = 0
        sigma_Y_tr = 0
        policy_pre = copy.deepcopy(policy)
        print('train step: {}'.format(iter_count))
        for n_index,state_val in enumerate(zip(X_s, X_next)):
            count += 1
            x_s, x_next = state_val
            val_n = []
            for i in range(3):
                val_n.append(x_next[i].T*w)
            max_index = val_n.index(max(val_n))
            # print(max_index)
            Y_t = x_s - gamma*x_next[max_index]
            policy[n_index] = max_index
            # print(policy[n_index],policy_pre[n_index])
            sigma_Y_t += Y_t*Y_t.T
            sigma_Y_tr +=  Y_t*R[n_index]
        w_pre = w
        w = sigma_Y_t.I*sigma_Y_tr
        if policy != policy_pre:
            nums = count_num(policy,policy_pre)
            print("diff num on policy:{}".format(nums))
        elif iter_count > 1:
            print("策略收敛!")
    np.save('lspi_15000_2.npy', w)


def count_num(l1,l2):
    res = 0
    for i,j in zip(l1,l2):
        if i!=j:
            res += 1
        else:
            continue
    return res

def display():
    w = np.load('lspi_15000_2.npy')
    env = Inverted_Pole()
    num_epoch = 1000
    max_steps = 700
    for epo_i in range(num_epoch):
        s = env.reset()
        for i in range(max_steps):
            val = []
            for p in range(3):
                val.append(x_state(s, p).T*w)
            index = np.argmax(val)
            s_next, reward, done_flag = env.step(index)
            # print(reward)
            s = s_next
            if done_flag == 0:
                break


def display_q(s_set_num):
    w = np.load('lspi_15000_2.npy')
    x = [-pi + i*2*pi/s_set_num for i in range(s_set_num)]
    y = [-15*pi + i*30*pi/s_set_num for i in range(s_set_num)]
    z = [[0 for i in range(s_set_num)] for j in range(s_set_num)]
    for i in range(s_set_num):
        for j in range(s_set_num):
            # print((x_state([-2*pi + i*4*pi/s_set_num,-15*pi + j*30*pi/s_set_num],1)*w.T)[0,0])
            z[i][j] = (x_state([-pi + i*2*pi/s_set_num,-15*pi + j*30*pi/s_set_num],1).T*w)[0,0]
    z = np.array(z)
    x, y = np.meshgrid(x, y)
    figure2 = plt.figure()
    ax = Axes3D(figure2)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)
    ax.set_xlabel('alpha', color='b')
    ax.set_ylabel('alpha v', color='r')
    ax.set_zlabel('Q(a, av)', color='g')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # train()

    from Inverted_Pole_env import Inverted_Pole # 展示用的环境
    display()
    
    # display_q(40)
