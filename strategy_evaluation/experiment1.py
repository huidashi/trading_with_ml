from fontTools.varLib.avarPlanner import measureSlant

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
import subprocess
import ManualStrategy as msl
import StrategyLearner as sl

def author():
  return 'hshi320'

def plot_out_graph(df,file_name_csv, insample=True, ):
    ax1 = plt.gca()

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax1.set_xlim([df.index.min(), df.index.max()])
    plt.xticks(rotation=90)

    ax1.plot(df.index, df['Normalized QLearn Portfolio Value'], color='red',
             label='Normalized QLearn Portfolio Value')
    ax1.plot(df.index, df['Normalized Benchmark Portfolio Value'], color='purple',
             label='Normalized Benchmark Portfolio Value')
    ax1.plot(df.index, df['Normalized Manual Strategy Portfolio Value'], color='green',
             label='Normalized Manual Strategy Portfolio Value')
    ax1.set_ylabel('Normalized Portfolio Value')
    ax1.tick_params(axis='y')


    # df['QLearn Orders Cumsum'] = df['QLearn Orders'].cumsum()
    # previous_cumsum = None
    #
    # for date, cumsum_value in df['QLearn Orders Cumsum'].items():
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


    custom_legend = [
        #Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Long'),
        #Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Short'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=1, label='Benchmark Portfolio Value'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='QLearn Portfolio Value'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='Manual Strategy Portfolio Value')
    ]
    # Display legends and title
    ax1.legend(loc='upper left', handles=custom_legend)
    if insample:
        plt.title('In-Sample QLearn & Manual & Benchmark Strategy')
    else:
        plt.title('Out-Sample QLearn & Manual & Benchmark Strategy')
    #plt.show()

    plt.savefig(file_name_csv)


if __name__ == "__main__":
    symbol = 'JPM'

    #insample_df
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    insample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
    insample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)

    learner = sl.StrategyLearner(verbose=True, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    qlearn_orders = learner.testPolicy(symbol=symbol, sd=sd, ed=ed,sv=100000)

    qlearn_orders['Symbol'] = symbol
    qlearn_orders_port_val = compute_portvals(df=qlearn_orders)
    qlearn_orders['QLearn Orders'] = qlearn_orders['Order']
    qlearn_orders_port_val.columns = ['QLearn Portfolio Value']

    benchmark_orders = qlearn_orders
    benchmark_orders['Order'] = 0
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.iloc[0, 0] = 1000
    benchmark_port_val = compute_portvals(df=benchmark_orders, start_val=100000, impact=0.005, commission=9.95)
    benchmark_port_val.columns = ['Benchmark Portfolio Value']


    manual_orders = msl.ManualStrategy().testPolicy(symbol=symbol, sd=sd, ed=ed)
    manual_orders['Symbol'] = symbol
    manual_strategy_port_val = compute_portvals(df=manual_orders)
    manual_orders['Manual Orders'] = manual_orders['Order']
    manual_strategy_port_val.columns = ['Manual Strategy Portfolio Value']


    insample_df = pd.concat(
        [insample_df, qlearn_orders_port_val, qlearn_orders, benchmark_port_val, manual_orders, manual_strategy_port_val], axis=1)

    plt.figure(figsize=(9, 6))
    insample_df[['Normalized QLearn Portfolio Value']] = insample_df[['QLearn Portfolio Value']].div(
        insample_df['QLearn Portfolio Value'].iloc[0])
    insample_df[['Normalized Benchmark Portfolio Value']] = insample_df[['Benchmark Portfolio Value']].div(
        insample_df['Benchmark Portfolio Value'].iloc[0])
    insample_df[['Normalized Manual Strategy Portfolio Value']] = insample_df[['Manual Strategy Portfolio Value']].div(
        insample_df['Manual Strategy Portfolio Value'].iloc[0])


    plot_out_graph(insample_df, insample=True, file_name_csv='in_sample_strategy.png')


    #outsample

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    outsample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
    outsample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)

    learner = sl.StrategyLearner(verbose=True, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    qlearn_orders = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                       sv=100000)

    qlearn_orders['Symbol'] = symbol
    qlearn_orders_port_val = compute_portvals(df=qlearn_orders)
    qlearn_orders['QLearn Orders'] = qlearn_orders['Order']
    qlearn_orders_port_val.columns = ['QLearn Portfolio Value']

    benchmark_orders = qlearn_orders
    benchmark_orders['Order'] = 0
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.iloc[0, 0] = 1000
    benchmark_port_val = compute_portvals(df=benchmark_orders, start_val=100000, impact=0.005, commission=9.95)
    benchmark_port_val.columns = ['Benchmark Portfolio Value']


    manual_orders = msl.ManualStrategy().testPolicy(symbol=symbol, sd=sd, ed=ed)
    manual_orders['Symbol'] = symbol
    manual_strategy_port_val = compute_portvals(df=manual_orders)
    manual_orders['Manual Orders'] = manual_orders['Order']
    manual_strategy_port_val.columns = ['Manual Strategy Portfolio Value']


    outsample_df = pd.concat(
        [outsample_df, qlearn_orders_port_val, qlearn_orders, benchmark_port_val, manual_orders, manual_strategy_port_val], axis=1)

    plt.figure(figsize=(9, 6))
    outsample_df[['Normalized QLearn Portfolio Value']] = outsample_df[['QLearn Portfolio Value']].div(
        outsample_df['QLearn Portfolio Value'].iloc[0])
    outsample_df[['Normalized Benchmark Portfolio Value']] = outsample_df[['Benchmark Portfolio Value']].div(
        outsample_df['Benchmark Portfolio Value'].iloc[0])
    outsample_df[['Normalized Manual Strategy Portfolio Value']] = outsample_df[['Manual Strategy Portfolio Value']].div(
        outsample_df['Manual Strategy Portfolio Value'].iloc[0])




    plot_out_graph(outsample_df, insample=False, file_name_csv='out_sample_strategy.png')
