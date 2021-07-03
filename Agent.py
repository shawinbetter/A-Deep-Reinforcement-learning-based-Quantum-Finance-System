#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qiu Yaowen
@file: Agent.py
@function: Create Class for transaction agent
@time: 2021/5/14 20:21
"""

from DeepQNetwork import DeepQNetwork
import numpy as np
import torch as T

class Agent():

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions=3,
                 max_mem_size=10000000, eps_end=1e-4, eps_dec=1e-4,fc1_dims=256, fc2_dims=128,fc3_dims=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, n_actions=self.n_actions,
                                   fc1_dims=fc1_dims, fc2_dims=fc2_dims,fc3_dims=fc3_dims)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.loss = None
    def save_model(self):
        T.save(self.Q_eval, 'DNN_Params')

    def load_model(self):
        self.Q_eval = T.load('DNN_Params')

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        #print("store_transition index:", index)

    # observation就是状态state
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:

            state = T.tensor(observation).to(self.Q_eval.device)

            # Get the return of network
            actions = self.Q_eval.forward(state)

            # print(actions.size())
            action = T.argmax(actions).item()
        else:
            # execute random behaviour with the probability of epsilon

            action = np.random.choice(self.action_space)
            #print("random action:", action)
        return action

    def learn(self):

        if self.mem_cntr < self.batch_size:
            #print("learn:watching")
            return

        # Initialize the gradient
        self.Q_eval.optimizer.zero_grad()

        # Return the size of memory
        max_mem = min(self.mem_cntr, self.mem_size)

        # Randomize a batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Array with index
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # extract a batch from the matrix
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]


        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.loss = self.Q_eval.loss(q_target, q_eval).detach().numpy()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min