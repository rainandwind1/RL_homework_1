import turtle as t
import os
import numpy as np
import math

# self.alpha= [-pi,pi]
# self.alpha_v = [-15*pi,15*pi]
# self.a = [-3,0,3]
# 采样时间 Ts = 0.005s

display = False
t.Turtle().screen.delay(0)
Q_rew = np.matrix([[5.,0.],[0.,0.1]])
R_rew = 1.0
T_s = 0.005
a = [-3,0,3]

pi_scale = 180/math.pi
pi = math.pi
t.setup(1000,1000)
t.pensize(5)
t.speed(10)
t.pencolor('purple')
map_scale = 5


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




class Inverted_Pole():
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
        
        bar = bar_init("Pole",140)
        self.bar = t.Turtle()
        self.bar.shape("Pole")
        self.bar.shapesize(1,1,11)
        # self.bar.setheading(-60)

        self.state = [self.alpha,self.alpha_v]  # 状态
        self.a = 0

    def init_params(self):
        self.alpha, self.alpha_v,self.alpha_av = -pi,0,0.0 # 起始状态 目标状态[0,0]
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
        pre_alpha = self.alpha
        pre_alpha_v = self.alpha_v
        self.alpha += T_s*self.alpha_v
        self.alpha_av = (1.0/self.J)*(self.m*self.g*self.l*math.sin(pre_alpha) - self.b*pre_alpha_v - self.K**2/self.R*pre_alpha_v + self.K/self.R*self.a)
        self.alpha_v += T_s*self.alpha_av
        self.is_vaild()
        
        self.render()
        self.state = [self.alpha,self.alpha_v]
        #print(self.alpha,self.alpha_av,self.alpha_v,self.a)

    def reset(self):
        t.goto(0,0)
        t.pendown()
        t.dot(25)
        t.pensize(5)
        t.penup()
        t.goto(-250,-250)
        t.pendown()
        t.goto(250,-250)
        t.goto(250,250)
        t.goto(-250,250)
        t.goto(-250,-250)
        t.penup()
        t.goto(0,0)
        self.init_params()
        #print(self.state)
        return self.state

    def render(self):
        if display:
            self.bar.speed(10)
            self.bar.setheading((pi-self.alpha)*pi_scale)

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


def main():
    env = Inverted_Pole()
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
    t.mainloop()




if __name__ == "__main__":
    main()



