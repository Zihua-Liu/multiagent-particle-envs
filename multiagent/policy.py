import numpy as np
from pyglet.window import key
import random

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# random move policy
class RandomPolicy(Policy):
    def __init__(self, env, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]

    def action(self, obs):
        # ignore observation and randomly make next move
        if self.env.discrete_action_input:
            u = random.randint(0, 4)
        else:
            u = np.zeros(5)
            index = random.randint(0, 4)
            u[index] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

# human knowledge policy for simple scenario
class HumanKnowledgePolicy(Policy):
    def __init__(self, env, agent_index):
        super(HumanKnowledgePolicy, self).__init__()
        self.env = env
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]

    def action(self, obs):
        # stupidly make next move based on observation
        move_or_not = random.randint(0, 1)
        if self.env.discrete_action_input:
            if move_or_not == 1 or (obs[2] == 0 and obs[3] == 0):
                u = 0
            elif obs[2] >= 0 and obs[3] >= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u = 1
                else:
                    u = 3
            elif obs[2] >= 0 and obs[3] <= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u = 1
                else:
                    u = 4
            elif obs[2] <= 0 and obs[3] >= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u = 2
                else:
                    u = 3
            elif obs[2] <= 0 and obs[3] <= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u = 2
                else:
                    u = 4
        else:
            u = np.zeros(5)
            if move_or_not == 1 or (obs[2] == 0 and obs[3] == 0):
                u[0] += 1.0
            elif obs[2] >= 0 and obs[3] >= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u[1] += 1.0
                else:
                    u[3] += 1.0
            elif obs[2] >= 0 and obs[3] <= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u[1] += 1.0
                else:
                    u[4] += 1.0
            elif obs[2] <= 0 and obs[3] >= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u[2] += 1.0
                else:
                    u[3] += 1.0
            elif obs[2] <= 0 and obs[3] <= 0:
                if np.abs(obs[2]) >= np.abs(obs[3]):
                    u[2] += 1.0
                else:
                    u[4] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
