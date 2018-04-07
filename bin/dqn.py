#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gym
from RL_brain import DeepQNetwork


from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
from multiagent.policy import RandomPolicy
from multiagent.policy import HumanKnowledgePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)

    # 定义使用 DQN 的算法
    RL = DeepQNetwork(n_actions=env.action_space[0].n,
                  n_features=env.observation_space[0].shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0008,)

    total_steps = 0
    for i_episode in range(30):

        # 获取回合 i_episode 第一个 observation
        observation = env.reset()
        observation = observation[0]
        ep_r = 0
        cnt = 0
        while True:
            print("i_episode: " + str(i_episode) + "    cnt: " + str(cnt))
            env.render()    # 刷新环境

            action = RL.choose_action(observation)  # 选行为
            action_onehot = np.zeros(env.action_space[0].n)
            action_onehot[action] += 1.0
            observation_, reward, done, info = env.step([action_onehot]) # 获取下一个 state
            observation_ = observation_[0]
            reward = reward[0]
            done = done[0]  

            # 保存这一组记忆
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > 1000:
                RL.learn()  # 学习

            ep_r += reward
            if total_steps > 1000 and cnt > 500:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1
            cnt += 1
    # 最后输出 cost 曲线
    RL.plot_cost() 

    # 用训练好的DQN去测试
    observation = env.reset()
    observation = observation[0]
    reward_dic = {}
    cnt = 0
    while True:
        if cnt > 1000:
            break
        env.render()
        cnt = cnt + 1
        action = RL.choose_action(observation)  # 选行为
        action_onehot = np.zeros(env.action_space[0].n)
        action_onehot[action] += 1.0
        observation_, reward, done, info = env.step([action_onehot]) # 获取下一个 state
        observation_ = observation_[0]
        reward = reward[0]
        done = done[0]  

        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        # 学习
        RL.learn()

        observation = observation_
        # display rewards        
        for agent in env.world.agents:
            if not agent.name in reward_dic:
                reward_dic[agent.name] = []
            reward_dic[agent.name].append(env._get_reward(agent))
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))
    for agent in env.world.agents:
        y = reward_dic[agent.name]
        x = np.linspace(1, len(y), len(y))
        plt.plot(x, y)
        plt.show()




