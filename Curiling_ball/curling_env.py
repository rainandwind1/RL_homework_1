# reduis = 1, m = 1, length = width = 100, 碰撞后 0.9v, 移动方向反射
# 输入action = [F_x,F_y] F = {+5,-5} a = {(5,5),(5,-5),(-5,5),(-5,-5)}
# action频率 10hz, 冰球环境运行时状态更新频率 100hz, 冰壶的空气阻力 0.005v
# 环境给的reward的频率 10hz, 大小为 -d_target, 最大化累积奖励，每隔30秒重置
# 冰壶的位置，目标位置， V_x0,V_y0 = [-10,10]  (Episodic unpdate environment)
# ZPP 
import numpy as np
import turtle as t

# 显示会使得训练变慢
dis_play_key = True

t.setup(1000,1000)
t.pensize(5)
t.speed(10)
t.pencolor('purple')
map_scale = 4



class curling_env():
    def __init__(self):
        super(curling_env,self).__init__()
        self.v_x = 0.0
        self.v_y = 0.0
        self.a_x = 0.0
        self.a_y = 0.0
        self.F_x = 0.0
        self.F_y = 0.0
        self.x = 0.0
        self.y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target = t.Turtle()
        self.ball = t.Turtle()
        self.a_dict = [(5,5),(5,-5),(-5,5),(-5,-5)]
        self.state = [self.x,self.y,self.a_x,self.a_y,self.v_x,self.v_y,self.target_x,self.target_y]
    
    # 初始化函数
    def init_env_param(self):
        self.v_x,self.v_y = np.random.uniform(-10,10,2)
        self.x,self.y,self.target_x,self.target_y = np.random.uniform(0,100,4)
        while (self.x,self.y) == (self.target_x,self.target_y):
            self.x,self.y,self.target_x,self.target_y = np.random.uniform(0,100,4)
        self.a_x,self.a_y,self.F_x,self.F_y = -0.005*self.v_x**2, -0.005*self.v_y**2, -0.005*self.v_x**2, -0.005*self.v_y**2

    # 状态更新函数
    def update_state(self,action):
        F_x,F_y = action
        for i in range(10): # 0.1秒内仿真了10次，每次0.01秒（仿真频率为100hz，动作执行频率是10hz） 
            self.render()
            self.F_x = F_x - 0.005*self.v_x**2
            self.F_y = F_y - 0.005*self.v_y**2
            self.a_x = self.F_x
            self.a_y = self.F_y
            self.v_x += self.a_x*0.01
            self.v_y += self.a_y*0.01 
            self.x += self.v_x*0.01 + 0.5*self.a_x*(0.01)**2
            self.y += self.v_y*0.01 + 0.5*self.a_y*(0.01)**2
            self.isValid()

    # 奖励函数
    def Reward(self):
        distance = ((self.x - self.target_x)**2 + (self.y - self.target_y)**2)**0.5
        # print(distance)
        return distance

    # 重置
    def reset(self):
        self.init_env_param()

        if dis_play_key:
            # 画面初始化
            t.goto(0,0)
            t.goto(0,400)
            t.goto(400,400)
            t.goto(400,0)
            t.goto(0,0)
            t.penup()
            t.hideturtle()
            # ball
            self.ball.reset()
            self.ball.penup()
            self.ball.setpos(self.x*map_scale,self.y*map_scale)
            self.ball.pendown()
            self.ball.pensize(3)
            self.ball.speed(10)
            # target
            self.target.reset()
            self.target.penup()
            self.target.setpos(self.target_x*map_scale,self.target_y*map_scale)
            self.target.dot(30,"blue")
            self.target.hideturtle()
        
        self.state = [self.x,self.y,self.a_x,self.a_y,self.v_x,self.v_y,self.target_x,self.target_y]
        obversation = self.state
        return obversation

    # step 函数
    def step(self,a_choice):
        action = self.a_dict[a_choice]
        self.update_state(action)
        self.state = [self.x,self.y,self.a_x,self.a_y,self.v_x,self.v_y,self.target_x,self.target_y]
        s_next = self.state
        reward = -self.Reward()
        if (self.x,self.y) == (self.target_x,self.target_y):
            done_flag = 0.0
            self.reset()
        else:
            done_flag = 1.0
        return s_next,reward,done_flag

    # 判断冰球是否越界（撞墙） 分成四种情况 上下左右
    def isValid(self):
        # 左
        if self.x <= 0:
            self.x = -self.x
            self.v_x = -self.v_x
            self.a_x = -self.a_x
            self.F_x = -self.F_x
        # 右
        if self.x >= 100:
            self.x = 200 - self.x
            self.v_x = -self.v_x
            self.a_x = -self.a_x
            self.F_x = -self.F_x
        # 上
        if self.y >= 100:
            self.y = 200 - self.y
            self.v_y = -self.v_y
            self.a_y = -self.a_y
            self.F_y = -self.F_y
        # 下
        if self.y <= 0:
            self.y = -self.y
            self.v_y = -self.v_y
            self.a_y = -self.a_y
            self.F_y = -self.F_y

    # 刷新显示
    def render(self):
        if dis_play_key:
            self.ball.goto(self.x*map_scale,self.y*map_scale)
        else:
            return

def main():
    env = curling_env()
    for epo_i in range(100):
        reward_sum = 0.0
        env.reset()
        for i in range(300):
            action = np.random.randint(4)
            # print(action)
            s_next,reward,done = env.step(action)
            reward_sum += reward
            print(i,reward,reward_sum)
            if done == 0:
                break
    t.mainloop()



if __name__ == "__main__":
    main()
