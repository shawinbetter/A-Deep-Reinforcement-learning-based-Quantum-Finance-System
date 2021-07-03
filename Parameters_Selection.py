#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qiu Yaowen
@file: Parameters_Selection.py
@function: Select the best combination of hyper-parameters for each financial product.
           The criteria is the average profit of last 10 training (epoch 247 - epoch 256)
@time: 2021/5/15 00:08
"""
from Environment import Environment
from Agent import Agent
from Standardization import Preprocessor
import pandas as pd
import numpy as np
import os

#Parameters need to be optimal
GAMMAs = [0.1,0.3,0.5,0.7,0.9]

BATCH_SIZEs = [64,128]
LRs = [1e-2,1e-3,1e-4]

#generate combination of parameters
set_of_params = []
for GAMMA in GAMMAs:
    for BATCH_SIZE in BATCH_SIZEs:
        for LR in LRs:
            set_of_params.append((GAMMA,BATCH_SIZE,LR))

#Fixed Parameters
EPSILON = 0.8 #Probability of random walking
N_ACTIONS = 3
EPS_END = 2e-2 #Min of EPSILON
NGAMES = 256 #Epochs
fc1_dims = 256
fc2_dims = 128
fc3_dims = 64

#financial products
Products = ['AUDUSD','AIRBUS','GOOGLE','USD100M1','XAUUSD']

#Used Features
base = ['OPEN','CLOSE','HIGH','LOW','VOLUME','QPL','QPL+','QPL-','MA5','MA21','RSI']

for product in Products:
    print("-------Training Model for %s --------" % product)
    data = pd.read_csv('Data/' + product + '/Source.csv')[base].iloc[::-1]
    data = Preprocessor(data).Get_preprocessed_data()
    # print(data)
    INPUT_DIMS = [len(data.columns) + 3]

    best_avg_profit = -1e9 # A extreme negative number to represent negative infinity
    best_params = None

    DIR = 'Parameters/parameters_selection_log_'+product+'.txt'
    if os.path.exists(DIR):
        os.remove(DIR)
    f = open(DIR,'w')

    for params in set_of_params:
        gamma,batch_size,lr = params
        print("Tranning for parameters set (%.1f, %f, %.4f)" %(gamma,batch_size,lr))
        f.write('GAMMA = %.2f BATCH_SIZE = %.1f LR = %.4f \n' %(gamma,batch_size,lr))
        env = Environment(data)
        agent = Agent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS, eps_end=EPS_END,
                      input_dims=INPUT_DIMS, lr=LR,
                      fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)
        profits,loss_history = [], []

        for i in range(NGAMES):
            profit = 0
            done = False
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action,1)
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
            loss_history.append(agent.loss)
            profits.append(env.total_profit)

        # print(str(params) + 'Succeed!')

        f.write('The avg profit of last 10 epochs is %.4f. The avg loss of last 10 epochs is %.4f \n' %(np.mean(profits[-10::]),np.mean(loss_history[-10::])))
        f.write('-----------------------------------\n')

        if np.mean(profits[-10::]) > best_avg_profit: #criteria
            best_params = params
            best_avg_profit = np.mean(profits[-10::])

    f.close()
    with open(DIR, 'r+') as f: #Write the optimal combination in the head of file
        content = f.read()
        f.seek(0, 0)
        f.write('Best Combination of Parameters is' + str(best_params) + '\n')
        f.write('The avg profit of last 10 epochs is %.4f. \n' % (best_avg_profit))
        f.write('******************************************** \n'+ content)
        f.close()









