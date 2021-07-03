"""
@author: Qiu Yaowen
@file: Environment.py
@function: Create Class for Transaction Environment
@time: 2021/5/14 20:21
"""

import math
import numpy as np

class Environment():
    def __init__(self,data):
        self.data = data
        self.barpos = 0

        self.buy_fee_rate = 0.0015
        self.sell_fee_rate = 0.0015

        self.init = 1e7
        self.fund = 1e7
        self.position = 0
        self.market_value = 0

        self.balance = 1e7
        self.total_profit = 0
        self.day_profit = 0

        self.order_size = 7e6 # To simplify the problem, set a fixed amount order_size percentage

    def reset(self):
        self.barpos = 0

        self.init = 1e7
        self.fund = 1e7
        self.position = 0
        self.market_value = 0

        self.balance = 1e7
        self.total_profit = 0
        self.day_profit = 0

        #The contruction of input data
        observation = list(self.data.iloc[self.barpos])
        observation.append(self.balance - self.init)
        observation.append(self.position)
        observation.append(self.fund)
        return (observation)

    def step(self, action, file, write=False):
        # action np.array([0,1,0])
        current_price = self.data['CLOSE'][self.barpos]
        if action == 0:  # BUY
            # if self.position != 0:
            #     if write and file != None:
            #         file.write("Trade day " + str(self.barpos+1) + " There are position in the account. Do Nothing.")
            #     pass
            if self.fund < self.order_size:
                if write and file != None:
                    file.write("Trade day " + str(self.barpos+1) + " Not enough fund to buy. Do Nothing.")
                pass
            else:
                buy_order = math.floor(self.order_size / self.data['CLOSE'][self.barpos]) # amount of shares
                self.position += buy_order
                trade_amount = buy_order * current_price # Actual transaction amount
                buy_fee = trade_amount * self.buy_fee_rate # buy fee to deduct
                self.fund = self.fund - trade_amount - buy_fee
                # print("Successfully Order %.2f  " %trade_amount)
                if write and file != None:
                    file.write("Trade day " + str(self.barpos+1) + " Successfully Order %.2f  . Buy Fee = %.2f." %(trade_amount,buy_fee))
        elif action == 1:  # SELL (MUST SELL ALL)
            if self.position > 0:
                sell_order = self.position
                self.position -= sell_order
                sell_fee = sell_order * current_price * self.sell_fee_rate
                trade_amount = sell_order * current_price
                self.fund = self.fund + trade_amount - sell_fee
                # print("Successfully Sell %.2f  " %trade_amount)
                if write and file != None:
                    file.write("Trade day " + str(self.barpos+1) + " Successfully Sell %.2f  . Sell Fee = %.2f." % (trade_amount,sell_fee))
            else:
                if write and file != None:
                    file.write("Trade day " + str(self.barpos+1) + " Not Enough Share to Sell. Do Nothing" )
                pass
        else:  # DO NOTHING
            if write:
                file.write("Trade day " + str(self.barpos+1) + " Keep." )
            # print("DO NOTHING")
            pass

        # Re-calculate the current financial situation
        self.market_value = self.position * current_price # the market value of positions
        self.yesterday_balance = self.balance #record
        self.balance = self.market_value + self.fund # the total account balance we have
        self.total_profit = self.balance - self.init # the floating profit and loss
        self.barpos += 1 #next trade day
        if write and file != None:
            file.write('The Current Balance is %.2f. The Current Fund is %.2f' %(self.balance,self.fund) + '\n')

        observation = list(self.data.iloc[self.barpos])
        observation.append(self.balance - self.init)
        observation.append(self.position)
        observation.append(self.fund)

        tomorrow_pct_change = self.data['PCT_CHANGE'][self.barpos]

        if tomorrow_pct_change == 0:
            reward = 0

        elif action == 0: #Today the action is to buy
            reward =   np.sign(tomorrow_pct_change) *np.log(abs(self.order_size * tomorrow_pct_change))

        elif action == 1 : #Today the action is to sell

            reward = np.sign(tomorrow_pct_change) * -5 * np.log(abs(self.order_size * tomorrow_pct_change))

        else: #Today the action is to keep
            if self.position == 0: #Have zero shares
                reward = -1 * np.sign(tomorrow_pct_change) * np.log(abs(self.order_size * tomorrow_pct_change))
            else: #Have positi-ve shares
                reward = np.sign(tomorrow_pct_change) * np.log(abs(self.position * current_price * tomorrow_pct_change))

        if self.barpos == len(self.data) - 1: # stop
            done = True
        else: # continue
            done = False
        return (observation, reward, done)