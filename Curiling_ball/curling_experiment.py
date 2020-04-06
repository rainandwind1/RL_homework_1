import numpy as np
import turtle as t
from curling_env import curling_env
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers,layers,losses
from DQN_tf import DQN,train,plot_curse
import os

LOAD_KEY = True
path = 'E:\Code\param\culing_dqn.ckpt'

t.setup(1000,1000)
t.pensize(5)
t.speed(10)
t.pencolor('purple')
map_scale = 4

# Hyperparameter
learning_rate = 0.001
memory_len = 10000
gamma = 0.9
batch_size = 64
output_size = 4
state_size = 8


epoch_num = 500
max_steps = 300
update_target_interval = 25
replay_time = 30

# 初始化
Q_value = DQN(output_size=output_size,memory_len = memory_len)
Q_target =  DQN(output_size=output_size,memory_len = memory_len)
Q_value.build(input_shape=(1,state_size))
Q_target.build(input_shape=(1,state_size))

if LOAD_KEY:
    Q_value.load_weights(path)
    Q_target.load_weights(path)
    print("Load weights!")

optimizer = optimizers.Adam(lr = learning_rate)
score_list = []
loss_list = []

def main():
    env = curling_env()
    train_flag = False
    score_avg = 0.0
    for epo_i in range(epoch_num):
        score = 0.0
        epsilon = max(0.01,0.1 - 0.01*(epo_i)/200)
        s = env.reset()
        for i in range(max_steps):
            action = Q_value.sample_action(s,epsilon)
            s_next,reward,done_flag = env.step(action)
            Q_value.save_memory((s,action,reward,s_next,done_flag))
            
            score += reward
            s = s_next
            # print(s_next)
            if done_flag == 0:
                break
        score_list.append(score)
        score_avg += score
        if len(Q_value.memory_list) >= 2000:
            train_flag = True
            train(Q_value,Q_target,optimizer,batch_size,gamma,loss_list,replay_time)
        if (epo_i+1) % update_target_interval == 0 and epo_i > 0:
            for raw,target in zip(Q_value.variables,Q_target.variables):
                target.assign(raw)
            score_avg = 0.0
        print("{} epoch score: {}, training: {}".format(epo_i+1,score,train_flag))       
    plot_curse(score_list,loss_list)
    Q_value.save_weights(path)
    #env.close()
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