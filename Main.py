"""
@author: Qiu Yaowen
@file: Main.py
@function: Main function to run
@time: 2021/5/14 20:21
"""

from Environment import Environment
from Agent import Agent
from Standardization import Preprocessor
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from Plot_K_Line_transaction import draw_transaction
from Financial_Tools import MaxDrawdown, SharpeRatio
import seaborn as sns
sns.set_style("white")

GAMMAs = [0.3,0.5,0.7,0.9,0.5]
BATCH_SIZEs = [128,64,128,128,64]
LRs = [0.001,0.001,0.001,0.001,0.001]


N_ACTIONS = 3

EPSILON = 0.9 #Probability of random walking
EPS_END = 5e-2 #Min of EPSILON

NGAMES = 256#Epochs

fc1_dims = 256
fc2_dims = 128
fc3_dims = 64

Products = ['AUDUSD','AIRBUS','GOOGLE','USD100M1','XAUUSD']
# Products = ['AUDUSD']

# Used Features
base = ['OPEN','CLOSE','HIGH','LOW','VOLUME','QPL','QPL+','QPL-','MA5','MA21','RSI']

# create an empty dictionary to store result for each product
dic = {}
for product in Products:
    dic[product] = []

if __name__ == '__main__':
    # data = pd.read_csv("Data/test_data.csv")
    # data = pd.read_csv('Data/AUDUSD/source.csv')

    # Create file for recording index of strategy
    DIR = 'Results/index.txt'
    if os.path.exists(DIR):
        os.remove(DIR)
    g = open(DIR,'w')

    for product in Products:
        if product in ['AUDUSD','USDJPY']:
            if_forex = True
        else:
            if_forex = False
        print("-------Training Model for %s --------" %product)
        data = pd.read_csv('Data/'+product+'/source.csv')[base].iloc[::-1].reset_index().iloc[:,1::]
        data = Preprocessor(data).Get_preprocessed_data()
        # print(data)
        INPUT_DIMS = [len(data.columns) + 3]

        env = Environment(data)
        GAMMA,BATCH_SIZE,LR = GAMMAs[Products.index(product)],BATCH_SIZEs[Products.index(product)],LRs[Products.index(product)]
        agent = Agent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS, eps_end=EPS_END, input_dims=INPUT_DIMS, lr=LR,
                      fc1_dims=fc1_dims, fc2_dims=fc2_dims,fc3_dims=fc3_dims)

        loss_history = []

        best_profit = -1e7
        best_log = 'Results/log/best_log_' + product + '.txt'
        best_balances = []
        for i in range(NGAMES):

            flag = True
            balances = []

            tmp_log = 'Results/log/tmp_log_' + product + '.txt'
            if os.path.exists(tmp_log):
                os.remove(tmp_log )
            file = open(tmp_log, 'w')

            done = False
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action,file,write = flag)
                if flag:
                    balances.append(env.balance)
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_

            loss_history.append(agent.loss)

            print('episode', i, ': profits %.2f' % env.total_profit,'MSE loss %.2f' % agent.loss)

            file.close()

            if env.total_profit >= best_profit: #if it is the best result ever
                shutil.copy(tmp_log, best_log)
                best_balances = balances
                best_profit = env.total_profit

        os.remove(tmp_log) #delete the temp log file

        g.write(product+":\n")
        g.write("The Total Profit is %.2f \n" % best_profit)
        g.write("The Accumulative Return Rate is the last trade day is %.2f" % (100* ((best_balances[-1] - 1e7) / 1e7)) + '% \n')
        g.write("The Maximum Return Rate is  is %.2f" % (100 * ((max(best_balances) - 1e7) / 1e7)) + '% \n')
        g.write("The Minimum Return Rate is  is %.2f" % (100 * ((min(best_balances) - 1e7) / 1e7)) + '% \n')
        # g.write("The Max Draw Down rate for %s is %.2f " %(product, 100*MaxDrawdown(best_balances)) + '% \n')
        # g.write("The Sharpe Ratio is %.2f " % (100*SharpeRatio(best_balances)) + '% \n')
        g.write('\n')

        #Plot each Transaction on the Kline
        draw_transaction(product)

        #store into dictionary
        dic[product].append(loss_history)
        dic[product].append(best_balances)

    g.close()

    #Reset ploting
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("white")
    x = [i + 1 for i in range(NGAMES)]

    # Plot loss curve
    plt.figure(figsize=[12, 8], dpi=200)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE Loss  in each Epoch')
    for product in Products:
        plt.plot(x, dic[product][0], label=product)
    plt.legend(loc='upper right')
    plt.savefig('Results/graph/loss_curve.png', dpi=200)

    # Plot balance curve for last round
    x = [i + 1 for i in range(len(data)-1)]
    y = [0 for i in range(len(x))]
    plt.figure(figsize=[12, 8], dpi=200)
    plt.xlabel('Trade days')
    plt.ylabel('Ratio')
    plt.title('Accumulative Return Rate')
    for product in Products:
        plt.plot(x, [(i-1e7)/1e7 for i in dic[product][1]], label=product)
    plt.plot(x,y,label='Principle')
    plt.legend(loc='upper left')
    plt.savefig('Results/graph/balance_curve.png', dpi=200)