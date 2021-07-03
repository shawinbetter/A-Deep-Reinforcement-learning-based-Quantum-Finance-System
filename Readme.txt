*****************************************************
A Reinforcement learning based Quantum Finance System
*****************************************************

*********************************************************************
The project aims to use deep reinforcement learning model to build an automatic trading system.

The dataset we used is 1000-days transaction data of 5 different financial products: 

Forex : AUD / USD
US Share: GOOGLE
CFD: AIRBUS
Future: USD 100 M1
Spot Gold: XAUUSD

The Environment (Features) we used are:

Transaction Data: [Open, High, Low, Close, Volume]
Indexes Data: [QPL, QPL-, QPL+, MA5,MA21, RSI]
Account Data: [Account Balance, Fund, Profit]
*********************************************************************

*********************************************************************
File Structure:
WorkingDirection/
├────.DS_Store
├────Agent.py
├────Data/
│    ├────.DS_Store
│    ├────AIRBUS/
│    │    ├────.DS_Store
│    │    └────Source.csv
│    ├────AUDUSD/
│    │    ├────.DS_Store
│    │    └────Source.csv
│    ├────GOOGLE/
│    │    ├────.DS_Store
│    │    └────Source.csv
│    ├────USD100M1/
│    │    ├────.DS_Store
│    │    └────Source.csv
│    └────XAUUSD/
│    │    ├────.DS_Store
│    │    └────Source.csv
├────DeepQNetwork.py #Create Network
├────Draw_File_Tree.py #Draw file tree
├────Environment.py #Create environment
├────Financial_Tools.py #Not used in the project
├────Main.py #Main entry
├────Parameters/ #Store the result of grid searching
│    ├────.DS_Store
│    ├────parameters_selection_log_AIRBUS.txt
│    ├────parameters_selection_log_AUDUSD.txt
│    ├────parameters_selection_log_GOOGLE.txt
│    ├────parameters_selection_log_USD100M1.txt
│    └────parameters_selection_log_XAUUSD.txt
├────Parameters_Selection.py #to select optimal combination of parameters
├────Plot_K_Line_origin.py #to plot k line without transaction
├────Plot_K_Line_transaction.py #to plot k line with transactions
├────project_input.mq4 #.mq4 file to retrieve data
├────Readme.txt
├────requirements.txt
├────Results/
│    ├────.DS_Store
│    ├────graph/
│    │    ├────.DS_Store
│    │    ├────balance_curve.png
│    │    ├────Kline_origin/
│    │    │    ├────.DS_Store
│    │    │    ├────AIRBUS_KLine.png
│    │    │    ├────AUDUSD_KLine.png
│    │    │    ├────GOOGLE_KLine.png
│    │    │    ├────USD100M1_KLine.png
│    │    │    └────XAUUSD_KLine.png
│    │    ├────Kline_transaction/
│    │    │    ├────.DS_Store
│    │    │    ├────AIRBUS_KLine.png
│    │    │    ├────AUDUSD_KLine.png
│    │    │    ├────GOOGLE_KLine.png
│    │    │    ├────USD100M1_KLine.png
│    │    │    └────XAUUSD_KLine.png
│    │    └────loss_curve.png
│    ├────index.txt
│    └────log/
│    │    ├────.DS_Store
│    │    ├────best_log_AIRBUS.txt
│    │    ├────best_log_AUDUSD.txt
│    │    ├────best_log_GOOGLE.txt
│    │    ├────best_log_USD100M1.txt
│    │    └────best_log_XAUUSD.txt
└────Standardization.py #to perform data preprocessing

*********************************************************************



