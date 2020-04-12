import numpy as np
a = [-3, 0, 3]

class pendulum_env(object):
    def __init__(self, ):
        self.m = 0.055
        self.g = 9.81
        self.l = 0.042
        self.j = 1.91e-4
        self.b = 3e-6
        self.k = 0.0536
        self.r = 9.5
        self.ts = 5e-3
        self.q_rew = np.array([[5, 0], [0, 0.1]])
        self.r_rew = 1
        self.state_size = 2
        self.action_size = 3

        self.theta = np.pi
        self.omega = 0

    def reset(self):
        self.theta = np.pi
        self.omega = 0
        return [self.theta, self.omega]

    def dynamic(self, theta, omega, action):
        return 1. / self.j * (self.m * self.g * self.l * np.sin(theta) -
                              self.b * omega - self.k**2 / self.r * omega +
                              self.k / self.r * action)

    def get_reward(self, action):
        s = np.array([[self.theta, self.omega]]).T
        return (-s.T.dot(self.q_rew).dot(s) - self.r_rew * action**2).item()

    def step(self, index):
        action = a[index]
        previous_theta = self.theta
        previous_omega = self.omega

        self.theta = previous_theta + self.ts * previous_omega
        self.omega = previous_omega + self.ts * self.dynamic(
            previous_theta, previous_omega, action)
        if self.theta >= np.pi:
            self.theta -= 2 * np.pi
        elif self.theta < -np.pi:
            self.theta += 2 * np.pi
        self.omega = np.clip(self.omega, -15 * np.pi, 15 * np.pi)
        reward = self.get_reward(action)
        if [self.theta, self.omega] == [0,0]:
            done_flag = 0
        else:
            done_flag = 1
        return [self.theta, self.omega], reward, done_flag