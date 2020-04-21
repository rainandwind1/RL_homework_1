# 1.将状态空间离散化成状态子集{Sk}
# 2.定义有限动作集Al
# 3.初始化Q(k,l)
# 4.repeat:
# 5.更新Q值
# 6. Q(k,l)_{i+1} = r + gamma*max(Q(k',l'))
# 7.i+=1
# 8.收敛
import copy
import turtle as t
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xzw_env import pendulum_env # 训练用的环境
# self.alpha= [-pi,pi]
# self.alpha_v = [-15*pi,15*pi]
# self.a = [-3,0,3]
# 采样时间 Ts = 0.005s


Q_rew = np.matrix([[5.,0.],[0.,0.1]])
R_rew = 1.0
T_s = 0.005
a = [-3,0,3]

pi_scale = 180/math.pi
pi = math.pi
# t.setup(1000,1000)
# t.pensize(5)
# t.speed(10)
# t.pencolor('purple')
# map_scale = 5


def random_walk():
    env = pendulum_env()
    for i in range(100):
        score = 0.0
        s = env.reset()
        for step in range(1000):
            action_choice = np.random.randint(3)
            s_next,reward,done_flag,info = env.step(action_choice)
            score += reward
            print(action_choice,s_next,reward)
            if done_flag == 0:
                print("break")
                break 
        print(i+1,score)
    # t.mainloop()

def Data_is_vaild(alpha, alpha_v):
    if alpha < -pi:
        alpha += 2*pi
    if alpha >= 3.14:
        alpha += -2*pi
    if alpha_v <= -15*pi:
        alpha_v = -15*pi
    if alpha_v >= 15*pi:
        alpha_v = 14.99*pi
    return alpha, alpha_v

def Reward(state, action):
    Q_rew = np.matrix([[5.,0.],[0.,0.1]])
    R_rew = 1.0
    reward = float(-np.matrix(state)*Q_rew*np.transpose(np.matrix(state))) - R_rew*action**2
    #print(self.state,self.alpha_av,reward)
    return reward

def plot_curse(target_list):
    figure1 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(target_list)):
        X.append(i)
    plt.plot(X,target_list,'-r')
    plt.xlabel('epoch')
    plt.ylabel('Q diff')
    plt.show()



s_set_num = 600 # 离散化尺度
def train():
    print("train beginning!")
    env = pendulum_env() 
    epoch_num = 1000
    delta = 1.0
    delta_min = 0.1
    gamma = 0.98
    Q_table = [[[0.0 for i in range(env.action_size)] for j in range(s_set_num)] for n in range(s_set_num)]
    iter_count = 0
    Q_diff = []
    a = [-3, 0, 3]
    while delta > delta_min:
        delta = 0.0
        Q_table_pre = copy.deepcopy(Q_table) 
        iter_count += 1
        for al in range(s_set_num):  # 角度
            for al_v in range(s_set_num):   # 角速度
                for j in range(env.action_size):

                    alpha = -pi + al*2*pi/s_set_num
                    alpha_v = -15*pi + al_v*30*pi/s_set_num
                    alpha_av = (1.0/env.j)*(env.m*env.g*env.l*math.sin(alpha) - env.b*alpha_v - env.k**2/env.r*alpha_v + env.k/env.r*a[j])
                    alpha_v_next = alpha_v + T_s*alpha_av
                    alpha_next = alpha + T_s*alpha_v
                    alpha_next, alpha_v_next = Data_is_vaild(alpha_next, alpha_v_next)
                    
                    al_next = int((alpha_next + pi)*s_set_num/(2*pi))
                    al_v_next = int((alpha_v_next + 15*pi)*s_set_num/(30*pi))
                    # print(alpha, alpha_v, alpha_av, alpha_next, alpha_v_next, al_next, al_v_next)
                    # print(alpha_next, al_next, al_v_next)
                    pre = Q_table[al][al_v][j]
                    Q_table[al][al_v][j] = Reward([alpha_next,alpha_v_next],a[j]) + gamma*max(Q_table_pre[al_next][al_v_next])
                    # print(Reward([alpha_next,alpha_v_next],a[j]))
                    delta = max(abs(Q_table[al][al_v][j] - Q_table_pre[al][al_v][j]),delta)
        print("{} iter: Q diff : {}".format(iter_count, delta))
        Q_diff.append(delta)
    Q_save = np.array(Q_table)
    np.save('Q_table.npy',Q_save) # 保存为.npy格式
    plot_curse(Q_diff)


def display():
    Q_table = np.load('Q_table.npy')
    Q_table = Q_table.tolist()
    # print(Q_table)
    num_epo = 1000
    env = Inverted_Pole()
    s = env.reset()
    max_step = 2000
    for epo_i in range(num_epo):
        s = env.reset()
        for i in range(max_step):
            alpha, alpha_v = s
            index_al = int((alpha + pi)*s_set_num/(2*pi))
            index_alv = int((alpha_v + 15*pi)*s_set_num/(30*pi))
            action = Q_table[index_al][index_alv].index(max(Q_table[index_al][index_alv]))
            print(alpha, alpha_v, action)
            s_next, reward, done_flag = env.step(action)
            # print(s_next,reward,action)
            s = s_next
            if done_flag == 0:
                print("finish!")
                break

def dis_q():
    Q_table = np.load('Q_table.npy')
    Q_table = Q_table.tolist()
    x = [-pi + i*2*pi/s_set_num for i in range(s_set_num)]
    y = [-15*pi + i*30*pi/s_set_num for i in range(s_set_num)]
    z = [[0 for i in range(s_set_num)] for j in range(s_set_num)]
    for i in range(s_set_num):
        for j in range(s_set_num):
            z[i][j] = Q_table[i][j][1]
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



train_flag = False
display_flag = False
if __name__ == "__main__":
    if train_flag:
        train()
    if display_flag:
        from Inverted_Pole_env import Inverted_Pole # 展示用的环境
        display()
    dis_q()






