""""""


"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: hshi320 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 904069365 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
import random  		  	   		 	   		  		  		    	 		 		   		 		  
import QLearner as ql
from util import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import indicators as indi
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from marketsimcode import compute_portvals

def author():
    return 'hshi320'
random.seed(904069365)
class StrategyLearner(object):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # constructor  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		 	   		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		 	   		  		  		    	 		 		   		 		  
        self.commission = commission
        self.num_bins = 10
        self.num_states = 1000

        self.learner = ql.QLearner(
                num_states=self.num_states,
                num_actions=3,
                alpha=0.19,
                gamma=0.90,
                rar=0.91,
                radr=0.99,
                dyna=0,
            verbose=False,
        )

  	#represent indiactors into bins for states
    def discretize(self,indicators):

        #create equal range bins and assign value to bin index
        bins = self.num_bins
        for indicator in indicators:
            bin_ranges=np.linspace(indicators[indicator].min(), indicators[indicator].max(), bins+1)
            indicators[f"{indicator}_state"] = np.digitize(indicators[indicator], bin_ranges, right=False) -1

        #combine
        indicators['state'] = indicators["bb_percent_state"] * bins**2 + indicators["stochastic_state"] * bins + indicators["sma_state"]
        # Scale the state to fit within num_states
        indicators["state"] = (indicators["state"] % self.num_states).astype(int)
        return indicators



    # this method should create a QLearner, and train it for trading  		  	   		 	   		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 1),
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		 	   		  		  		    	 		 		   		 		  
        df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
        df.rename(columns={symbol: 'Adj Close'}, inplace=True)
        df['Close'] = get_data([symbol], pd.date_range(sd, ed), colname='Close')[[symbol]]
        adjusted_ratio = df['Adj Close'] / df['Close']

        df['High'] = get_data([symbol], pd.date_range(sd, ed), colname='High')[[symbol]]
        df['High'] = df['High'] * adjusted_ratio
        df['Low'] = get_data([symbol], pd.date_range(sd, ed), colname='Low')[[symbol]]
        df['Low'] = df['Low'] * adjusted_ratio

        df['Bollinger Bands Percent'] = indi.calculate_bollinger_bands_percent(df['Adj Close'])
        df['Stochastic Oscillator'] = indi.calculate_stochastic_oscillator(df['Adj Close'], high_series=df['High'], low_series=df['Low'])
        df['SMA20'] = indi.calculate_SMA20(df['Adj Close'])

        indicators = df[['Bollinger Bands Percent', 'Stochastic Oscillator', 'SMA20']].copy()
        indicators.columns = ['bb_percent','stochastic', 'sma']
        indicators = self.discretize(indicators)

        #train
        states= indicators['state'].astype(int).values
        actions = np.zeros(len(states))
        rewards = np.zeros(len(states))

        current_position = 0  # +1 for long, -1 for short
        for epoch in range(500):
            total_reward = 0

            for i in range(1, len(states)):
                price_change = df['Adj Close'].iloc[i] - df['Adj Close'].iloc[i - 1]
                action = self.learner.querysetstate(states[i - 1])

                if action == 1:  # buy
                    new_position = 1
                elif action == 2:  # sell
                    new_position = -1
                else:
                    new_position = 0

                reward = current_position * price_change
                if new_position != current_position:
                    trade_size = abs(new_position - current_position)
                    impact_cost = self.impact * trade_size
                    commission_cost = self.commission
                    reward = reward - (impact_cost + commission_cost)

                action = self.learner.query(states[i], reward)

                total_reward = total_reward + reward
                current_position = new_position

            # early stop
            if epoch > 1 and abs(total_reward - np.sum(rewards)) < 1e-6:
                break

            rewards[:] = total_reward




    #
    def testPolicy(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
        """

        #getdata

        df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
        df.rename(columns={symbol: 'Adj Close'}, inplace=True)
        df['Close'] = get_data([symbol], pd.date_range(sd, ed), colname='Close')[[symbol]]
        adjusted_ratio = df['Adj Close'] / df['Close']

        df['High'] = get_data([symbol], pd.date_range(sd, ed), colname='High')[[symbol]]
        df['High'] = df['High'] * adjusted_ratio
        df['Low'] = get_data([symbol], pd.date_range(sd, ed), colname='Low')[[symbol]]
        df['Low'] = df['Low'] * adjusted_ratio

        df['Bollinger Bands Percent'] = indi.calculate_bollinger_bands_percent(df['Adj Close'])
        df['Stochastic Oscillator'] = indi.calculate_stochastic_oscillator(df['Adj Close'], high_series=df['High'],
                                                                           low_series=df['Low'])
        df['SMA20'] = indi.calculate_SMA20(df['Adj Close'])

        indicators = df[["Bollinger Bands Percent", "Stochastic Oscillator", "SMA20"]].copy()
        indicators.columns = ["bb_percent", "stochastic", "sma"]
        indicators = self.discretize(indicators)

        # test policy
        states = indicators["state"].astype(int).values
        trades = pd.DataFrame(data=0, index=df.index, columns=["Order"])

        #holding constraints
        current_holding = 0
        for i in range(len(states)):
            action = self.learner.query(states[i], testing=True)

            if action == 1:  # buy
                trade_amount = 1000 - current_holding
                trades.iloc[i] = trade_amount
                current_holding = 1000
            elif action == 2:  # sell
                trade_amount = -1000 - current_holding
                trades.iloc[i] = trade_amount
                current_holding = -1000
            else:  # HOLD
                trades.iloc[i] = 0  #holding stays neutral

        return trades
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    pass

    # symbol = 'JPM'
    #
    # sd = dt.datetime(2008, 1, 1)
    # ed = dt.datetime(2009, 12, 31)
    # outsample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
    # outsample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)
    #
    #
    # learner = StrategyLearner(verbose=True, impact=0.005, commission=9.95)
    # learner.add_evidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
    # qlearn_orders = learner.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
    #
    #
    #
    # qlearn_orders['Symbol'] = symbol
    # qlearn_orders_port_val = compute_portvals(df=qlearn_orders)
    # qlearn_orders['QLearn Orders'] = qlearn_orders['Order']
    # qlearn_orders_port_val.columns = ['QLearn Portfolio Value']
    #
    # benchmark_orders = qlearn_orders
    # benchmark_orders['Order'] = 0
    # benchmark_orders['Symbol'] = symbol
    # benchmark_orders.iloc[0, 0] = 1000
    # benchmark_port_val = compute_portvals(df=benchmark_orders, start_val=100000, impact=0.005, commission=9.95)
    # benchmark_port_val.columns = ['Benchmark Portfolio Value']
    #
    #
    #
    # outsample_df = pd.concat(
    #     [outsample_df, qlearn_orders_port_val,qlearn_orders,benchmark_port_val], axis=1)
    #
    # plt.figure(figsize=(9, 6))
    # outsample_df[['Normalized QLearn Portfolio Value']] = outsample_df[['QLearn Portfolio Value']].div(
    #     outsample_df['QLearn Portfolio Value'].iloc[0])
    # outsample_df[['Normalized Benchmark Portfolio Value']] = outsample_df[['Benchmark Portfolio Value']].div(
    #     outsample_df['Benchmark Portfolio Value'].iloc[0])
    #
    # ax1 = plt.gca()
    #
    # ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    # ax1.set_xlim([outsample_df.index.min(), outsample_df.index.max()])
    # plt.xticks(rotation=90)
    #
    # ax1.plot(outsample_df.index, outsample_df['Normalized QLearn Portfolio Value'], color='red',
    #          label='Normalized QLearn Portfolio Value')
    # ax1.plot(outsample_df.index, outsample_df['Normalized Benchmark Portfolio Value'], color='purple',
    #          label='Normalized Benchmark Portfolio Value')
    # ax1.set_ylabel('Normalized Portfolio Value')
    # ax1.tick_params(axis='y')
    #
    # # loop through dates and cumulative sum values to plot lines based on conditions
    # outsample_df['QLearn Orders Cumsum'] = outsample_df['QLearn Orders'].cumsum()
    # previous_cumsum = None
    #
    # for date, cumsum_value in outsample_df['QLearn Orders Cumsum'].items():
    #     # Cumsum turns pos
    #     if cumsum_value > 0 and (previous_cumsum is None or previous_cumsum <= 0):
    #         ax1.axvline(date, color='blue', linestyle='--', linewidth=1)
    #     # Cumsum turns neg
    #     elif cumsum_value < 0 and (previous_cumsum is None or previous_cumsum >= 0):
    #
    #         ax1.axvline(date, color='black', linestyle='--', linewidth=1)
    #
    #     # Update the previous cumsum value
    #     previous_cumsum = cumsum_value
    #
    # custom_legend = [
    #     Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Long'),
    #     Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Short'),
    #     Line2D([0], [0], color='purple', linestyle='--', linewidth=1, label='Benchmark Portfolio Value'),
    #     Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Manual Strategy Portfolio Value')
    # ]
    # # Display legends and title
    # ax1.legend(loc='upper left', handles=custom_legend)
    # plt.title('Out-Sample QLearn vs Benchmark Strategy')
    # plt.show()
    #
    # outsample_df.to_csv('/Users/davidshi/Documents/ML4T_2024Fall/strategy_evaluation/outsample.csv')
    # print(outsample_df[['QLearn Portfolio Value']])
    print("One does not simply think up a strategy")