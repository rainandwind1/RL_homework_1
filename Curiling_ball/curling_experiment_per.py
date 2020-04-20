import numpy as np
import turtle as t
from curling_env import curling_env
from DQN_Prioritized_Replay_py import DQN, Replay_buffer, train, plot_curse
import os
import torch 
from torch import nn, optim

LOAD_KEY = True
path = 'param\culing_dqnper.pkl'

t.setup(1000,1000)
t.pensize(5)
t.speed(10)
t.pencolor('purple')
map_scale = 4

# Hyperparameter
learning_rate = 0.001
memory_len = 30000
gamma = 0.9
batch_size = 128
output_size = 4
state_size = 8
replay_len = 2000

epoch_num = 600
max_steps = 300
update_target_interval = 25
replay_time = 50
alpha = 0.6
beta = 0.4


# 初始化
Q_value = DQN(input_size = state_size, output_size=output_size,memory_len = memory_len)
Q_target = DQN(input_size = state_size, output_size=output_size,memory_len = memory_len)
replay_buffer = Replay_buffer(alpha,beta,memory_len)

optimizer = optim.Adam(Q_value.parameters(),lr = learning_rate)
loss = nn.MSELoss()
score_list = []
loss_list = []

if LOAD_KEY:
    checkpoint = torch.load(path)
    Q_value.load_state_dict(checkpoint)
    Q_target.load_state_dict(checkpoint)
    print("Load weights!")
else:
    print("No exist weights to use!")




def main():
    env = curling_env()
    score_avg = 0.0
    train_flag = False
    for epo_i in range(epoch_num):
        score = 0.0
        epsilon = max(0.01,0.1 - 0.01*(epo_i)/200)
        s = env.reset()
        for i in range(max_steps):
            action = Q_value.sample_action(s,epsilon)
            s_next,reward,done_flag = env.step(action)
            replay_buffer.store_transition((s,action,reward,s_next,done_flag))
            score += reward
            s = s_next
            # print(s_next)
            if done_flag == 0:
                break
        score_list.append(score)
        score_avg += score
        if replay_buffer.len >= replay_len:
            train_flag = True
            train(Q_value, Q_target, replay_buffer, optimizer, batch_size, loss, gamma, loss_list, Replay_time=20)
        if (epo_i+1) % update_target_interval == 0 and i > 0:
            Q_target.load_state_dict(Q_value.state_dict())
            print("%d epoch avg score: %d \n"%(epo_i+1,score_avg/update_target_interval))
            score_avg = 0.0
        print("{} epochs score: {}, training: {}".format(epo_i+1,score,train_flag)) 
    plot_curse(score_list,loss_list)
    # state = {‘net':Q_value.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(Q_value.state_dict(), path)
    t.mainloop()


if __name__ == "__main__":
    main()














# # 画笔参数
# t.setup(1000,1000)
# t.pensize(3)
# t.speed(1)
# t.pencolor('purple')

# def init_target(name,color,redius):
#     t.begin_poly()
#     t.dot(redius,color)  
#     t.end_poly()
#     shape = t.get_poly()
#     t.register_shape(name,shape)

# t.goto(0,0)
# t.goto(0,500)
# t.goto(500,500)
# t.goto(500,0)
# t.goto(0,0)
# t.penup()
# t.setpos(20,20)


# target = t.Turtle()
# target.penup()
# target.setpos(100,100)
# target.dot(30,"blue")
# target.reset()
# target.setpos(10,20)
# t.mainloop()