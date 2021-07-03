#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qiu Yaowen
@file: Standardization.py
@function: Perform necessary preprocessing on input data
@time: 2021/5/14 20:21
"""

import pandas as pd


class Preprocessor():
    def __init__(self, data):
        self.data = data

    def Add_pct_change(self):
        self.data['PCT_CHANGE'] = pd.Series(self.data['CLOSE']).pct_change() # get gaily change of price
        self.data = self.data.fillna(0) #the first row of pct_change will be zero

    def Standardization(self):
        # the scale of volume is too big
        self.data['VOLUME'] = (self.data['VOLUME']-self.data['VOLUME'].min())/(self.data['VOLUME'].max()-self.data['VOLUME'].min())

    def Get_preprocessed_data(self):
        self.Add_pct_change()
        self.Standardization()
        return self.data