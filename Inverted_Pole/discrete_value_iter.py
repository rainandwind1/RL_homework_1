# 1.将状态空间离散化成状态子集{Sk}
# 2.定义有限动作集Al
# 3.初始化Q(k,l)
# 4.repeat:
# 5.更新Q值
# 6. Q(k,l)_{i+1} = r + gamma*max(Q(k',l'))
# 7.i+=1
# 8.收敛

import turtle as t
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Inverted_Pole_env import Inverted_Pole

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


def Skip(step):
    t.penup()
    t.forward(step)
    t.pendown()



def bar_init(name, length):
    # 注册Turtle形状，建立表针Turtle
    # Skip(-length * 0.1)
    # 开始记录多边形的顶点。当前的乌龟位置是多边形的第一个顶点。
    t.begin_poly()
    t.penup()
    t.pensize(10)
    t.forward(length * 1.1)
    # 停止记录多边形的顶点。当前的乌龟位置是多边形的最后一个顶点。将与第一个顶点相连。
    t.end_poly()
    # 返回最后记录的多边形。
    handForm = t.get_poly()
    t.hideturtle()
    t.register_shape(name, handForm)




class Inverted_Pole_train():
    def __init__(self):
        self.m = 0.055      # 重量
        self.g = 9.81       # 重力加速度
        self.l = 0.042      # 重心到转子的距离
        self.J = 1.91*1e-4  # 转动惯量
        self.b = 3*1e-6     # 粘滞阻尼
        self.K = 0.0536     # 转矩常数
        self.R = 9.5        # 转子电阻
        self.state_size = 2
        self.action_size = 3

        self.alpha = 0.0    # 摆角
        self.alpha_v = 0.0  # 角速度
        self.alpha_av = 0.0 # 角加速度
        
        # bar = bar_init("Pole",140)
        # self.bar = t.Turtle()
        # self.bar.shape("Pole")
        # self.bar.shapesize(1,1,11)
        # self.bar.setheading(-60)

        self.state = [self.alpha,self.alpha_v]  # 状态
        self.a = 0

    def init_params(self):
        self.alpha, self.alpha_v,self.alpha_av = -pi,0.0,0.0 # 起始状态 目标状态[0,0]
        self.state = [self.alpha,self.alpha_v]
        

    def is_vaild(self):
        # print("is_valid")
        if self.alpha < -pi:
            self.alpha += 2*pi
        if self.alpha >= pi:
            self.alpha += -2*pi
        if self.alpha_v <= -15*pi:
            self.alpha_v = -15*pi
        if self.alpha_v >= 15*pi:
            self.alpha_v = 15*pi

    def Reward(self):
        self.state = [self.alpha,self.alpha_v]
        reward = float(-np.matrix(self.state)*Q_rew*np.transpose(np.matrix(self.state))) - R_rew*self.a**2
        #print(self.state,self.alpha_av,reward)
        return reward

    def update_param(self):
        self.alpha_av = (1.0/self.J)*(self.m*self.g*self.l*math.sin(self.alpha) - self.b*self.alpha_v - self.K**2/self.R*self.alpha_v + self.K/self.R*self.a)
        self.alpha_v += T_s*self.alpha_av
        self.alpha += T_s*self.alpha_v
        self.is_vaild()
        self.render()
        self.state = [self.alpha,self.alpha_v]
        #print(self.alpha,self.alpha_av,self.alpha_v,self.a)

    def reset(self):
        # t.goto(0,0)
        # t.pendown()
        # t.dot(25)
        # t.pensize(5)
        # t.penup()
        # t.goto(-250,-250)
        # t.pendown()
        # t.goto(250,-250)
        # t.goto(250,250)
        # t.goto(-250,250)
        # t.goto(-250,-250)
        # t.penup()
        # t.goto(0,0)
        self.init_params()
        #print(self.state)
        return self.state

    def render(self):
        return
        # self.bar.speed(30)
        # self.bar.setheading((pi-self.alpha)*pi_scale)

    def step(self,action_index):
        self.a = a[action_index]
        self.update_param()
        reward = self.Reward()
        # print(reward)
        s_next = self.state
        if s_next == [0,0]:
            done_flag = 0
        else:
            done_flag = 1
        return s_next,reward,done_flag


def random_walk():
    env = Inverted_Pole_train()
    for i in range(100):
        score = 0.0
        s = env.reset()
        for step in range(1000):
            action_choice = np.random.randint(3)
            s_next,reward,done_flag = env.step(action_choice)
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
    if alpha >= pi:
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



s_set_num = 200  # 离散化尺度
def train():
    print("train beginning!")
    env = Inverted_Pole_train() 
    epoch_num = 1000
    delta = 1.0
    delta_min = 1e-2
    gamma = 0.98
    Q_table = [[[0.0 for i in range(env.action_size)] for j in range(s_set_num)] for n in range(s_set_num)]
    iter_count = 0
    Q_diff = []
    a = [-3, 0, 3]
    while delta > delta_min:
        delta = 0.0
        iter_count += 1
        for al in range(s_set_num):  # 角度
            for al_v in range(s_set_num):   # 角速度
                for j in range(env.action_size):
                    alpha = -2*pi + al*4*pi/s_set_num
                    alpha_v = -15*pi + al_v*30*pi/s_set_num
                    alpha_av = (1.0/env.J)*(env.m*env.g*env.l*math.sin(alpha) - env.b*alpha_v - env.K**2/env.R*alpha_v + env.K/env.R*a[j])
                    alpha_v_next = alpha_v + T_s*alpha_av
                    alpha_next = alpha + T_s*alpha_v_next
                    alpha_next, alpha_v_next = Data_is_vaild(alpha_next, alpha_v_next)
                    
                    al_next = int((alpha_next + 2*pi)*s_set_num/(4*pi))
                    al_v_next = int((alpha_v_next + 15*pi)*s_set_num/(30*pi))
                    # print(alpha, alpha_v, alpha_av, alpha_next, alpha_v_next, al_next, al_v_next)
                    # print(al_next, al_v_next)
                    pre = Q_table[al][al_v][j]
                    Q_table[al][al_v][j] = Reward([alpha,alpha_av],a[j]) + gamma*max(Q_table[al_next][al_v_next])
                    delta += abs(Q_table[al][al_v][j] - pre)

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
    max_step = 800
    for epo_i in range(num_epo):
        s = env.reset()
        for i in range(max_step):
            alpha, alpha_v = s
            index_al = int((alpha + 2*pi)*s_set_num/(4*pi))
            index_alv = int((alpha_v + 15*pi)*s_set_num/(30*pi))
            action = Q_table[index_al][index_alv].index(max(Q_table[index_al][index_alv]))
            # print(alpha, alpha_v, action)
            s_next, reward, done_flag = env.step(action)
            # print(s_next,reward,action)
            s = s_next
            if done_flag == 0:
                print("finish!")
                break

def dis_q():
    Q_table = np.load('Q_table.npy')
    Q_table = Q_table.tolist()
    x = [i for i in range(s_set_num)]
    y = [i for i in range(s_set_num)]
    z = [[0 for i in range(s_set_num)] for j in range(s_set_num)]
    for i in range(s_set_num):
        for j in range(s_set_num):
            z[i][j] = max(Q_table[i][j])
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
display_flag = True
if __name__ == "__main__":
    if train_flag:
        train()
    if display_flag:
        display()
    # dis_q()






