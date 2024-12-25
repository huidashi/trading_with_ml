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
import math as m


def author():
    return 'hshi320'


def plot_out_graph(df, file_name_csv, insample=True, ):
    ax1 = plt.gca()

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax1.set_xlim([df.index.min(), df.index.max()])
    plt.xticks(rotation=90)

    ax1.plot(df.index, df['Normalized 0.005_impact_port_val'], color='red',
             label='Normalized QLearn 0.005_impact_port_val')
    ax1.plot(df.index, df['Normalized 0.05_impact_port_val'], color='purple',
             label='Normalized QLearn 0.05_impact_port_val')
    ax1.plot(df.index, df['Normalized 0.5_impact_port_val'], color='green',
             label='Normalized QLearn 0.5_impact_port_val')
    ax1.set_ylabel('Normalized Portfolio Value')
    ax1.tick_params(axis='y')



    custom_legend = [
        # Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Long'),
        # Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Short'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=1, label='Normalized QLearn 0.05_impact_port_val'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Normalized QLearn 0.005_impact_port_val'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='Normalized QLearn 0.5_impact_port_val')
    ]
    # Display legends and title
    ax1.legend(loc='upper left', handles=custom_legend)
    if insample == 'impact':
        plt.title('In-Sample QLearn Strategy with Different Impacts')
    elif insample:
        plt.title('In-Sample QLearn & Manual & Benchmark Strategy')
    else:
        plt.title('Out-Sample QLearn & Manual & Benchmark Strategy')
    #plt.show()

    plt.savefig(file_name_csv)

def calculate_metrics(port_val):
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]

    adr = daily_rets.mean()
    sddr = daily_rets.std()

    sr = (adr - 0) / sddr
    sr = (sr * m.sqrt(252))

    cr = (port_val[-1] / port_val[0]) - 1

    return ["{:.6f}".format(cr), round(sddr, 6), round(adr, 6)]

if __name__ == "__main__":
    symbol = 'JPM'
    impacts = [0.005,0.05,0.5]
    insample_df_full = pd.DataFrame()
    for impact in impacts:
        # insample_df
        sd = dt.datetime(2008, 1, 1)
        ed = dt.datetime(2009, 12, 31)
        insample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
        insample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)

        learner = sl.StrategyLearner(verbose=True, impact=impact, commission=9.95)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
        qlearn_orders = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)

        qlearn_orders['Symbol'] = symbol
        qlearn_orders_port_val = compute_portvals(df=qlearn_orders)
        qlearn_orders['QLearn Orders'] = qlearn_orders['Order']
        qlearn_orders_port_val.columns = [f"{impact}_impact_port_val"]

        metric_df = pd.DataFrame(
            columns=['Strategy', 'Cumulative Return', 'Standard Deviation Daily Returns', 'Average Daily Returns'])
        # metric_df['Strategy'] = ['Optimal Portfolio','Benchmark Portfolio']

        temp = f"{impact}_impact_port_val"
        manual_metrics = calculate_metrics(qlearn_orders_port_val[temp])
        manual_metrics.insert(0, f"{impact}_impact_port_val")

        metric_df.loc[len(metric_df)] = manual_metrics
        metric_df.to_csv(f"In Sample Strategy_Metric_Table_{impact}_impact_port_val.csv", index=False)

        insample_df_full = pd.concat(
            [insample_df_full,insample_df, qlearn_orders_port_val], axis=1)

        plt.figure(figsize=(9, 6))
        insample_df_full[[f"Normalized {impact}_impact_port_val"]] = insample_df_full[[f"{impact}_impact_port_val"]].div(insample_df_full[f"{impact}_impact_port_val"].iloc[0])


    plot_out_graph(insample_df_full, insample='impact', file_name_csv="in_sample_strategy_impact.png")