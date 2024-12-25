import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
from marketsimcode import compute_portvals
import indicators as indi
import datetime as dt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import math as m

def author():
    return 'hshi320'

class ManualStrategy:
    def __init__(self, impact=0.005, commission=9.95, verbose=False):
        self.impact=impact
        self.commission=commission
        self.verbose=verbose

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):

        #get data
        df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
        df.rename(columns={symbol: 'Adj Close'}, inplace=True)
        df['Close'] = get_data([symbol], pd.date_range(sd, ed), colname='Close')[[symbol]]
        adjusted_ratio = df['Adj Close'] / df['Close']

        df['High'] = get_data([symbol], pd.date_range(sd, ed), colname='High')[[symbol]]
        df['High'] = df['High'] * adjusted_ratio
        df['Low'] = get_data([symbol], pd.date_range(sd, ed), colname='Low')[[symbol]]
        df['Low'] = df['Low'] * adjusted_ratio

        df['Bollinger Bands Percent'] = indi.calculate_bollinger_bands_percent(df['Adj Close'])
        df['Stochastic Oscillator'] = indi.calculate_stochastic_oscillator(df['Adj Close'], high_series=df['High'],low_series=df['Low'])
        df['SMA20'] = indi.calculate_SMA20(df['Adj Close'])

        #manual strategy
        df['Order'] = 0
        current_holding = 0

        for i in range(len(df)):
            #long
            if (df['Adj Close'].iloc[i] < df['SMA20'].iloc[i] and
                df['Bollinger Bands Percent'].iloc[i] < 15 and
                df['Stochastic Oscillator'].iloc[i] < 15):
                if current_holding <= 0:
                    df.loc[df.index[i], 'Order'] = 1000 - current_holding
                    current_holding = 1000

            # short signal
            elif (df['Adj Close'].iloc[i] > df['SMA20'].iloc[i] and
                  df['Bollinger Bands Percent'].iloc[i] > 85 and
                  df['Stochastic Oscillator'].iloc[i] > 85):
                if current_holding >= 0:
                    df.loc[df.index[i], 'Order'] = -1000 - current_holding
                    current_holding = -1000



        df['Symbol'] = symbol
        return df[['Order']]


    def calculate_metrics(self,port_val):
        daily_rets = (port_val / port_val.shift(1)) - 1
        daily_rets = daily_rets[1:]

        adr = daily_rets.mean()
        sddr = daily_rets.std()

        sr = (adr - 0) / sddr
        sr = (sr * m.sqrt(252))

        cr = (port_val[-1] / port_val[0]) - 1

        return ["{:.6f}".format(cr), round(sddr, 6), round(adr, 6)]

if __name__ == "__main__":
    strategy = ManualStrategy()

    #get data df
    symbol='JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    insample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
    insample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)

    #in sample manual Strategy
    manual_orders = strategy.testPolicy(sd=sd, ed=ed)

    manual_orders['Symbol'] = symbol
    manual_strategy_port_val = compute_portvals(df=manual_orders)
    manual_orders['Manual Orders'] = manual_orders['Order']
    manual_strategy_port_val.columns = ['Manual Strategy Portfolio Value']

    #in sample Benchmark Strategy
    benchmark_orders = manual_orders
    benchmark_orders['Order'] = 0
    benchmark_orders.iloc[0,0] = 1000
    benchmark_orders['Symbol'] = symbol
    benchmark_port_val = compute_portvals(df=benchmark_orders, start_val=100000, impact=0.005, commission=9.95)
    benchmark_port_val.columns = ['Benchmark Portfolio Value']

    insample_df = pd.concat([insample_df, benchmark_port_val, manual_strategy_port_val,manual_orders[['Manual Orders']]], axis=1)


    #in sample Strategy plot
    plt.figure(figsize=(9, 6))
    insample_df[['Normalized Manual Strategy Portfolio Value']] = insample_df[['Manual Strategy Portfolio Value']].div(insample_df['Manual Strategy Portfolio Value'].iloc[0])
    insample_df[['Normalized Benchmark Portfolio Value']] = insample_df[['Benchmark Portfolio Value']].div(insample_df['Benchmark Portfolio Value'].iloc[0])

    ax1 = plt.gca()

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax1.set_xlim([insample_df.index.min(), insample_df.index.max()])
    plt.xticks(rotation=90)

    ax1.plot(insample_df.index, insample_df['Normalized Manual Strategy Portfolio Value'], color='red',
                        label='Normalized Manual Strategy Portfolio Value')
    ax1.plot(insample_df.index, insample_df['Normalized Benchmark Portfolio Value'], color='purple',
                        label='Normalized Benchmark Portfolio Value')
    ax1.set_ylabel('Normalized Portfolio Value')
    ax1.tick_params(axis='y')


    # Loop through dates and cumulative sum values to plot lines based on conditions
    insample_df['Manual Orders Cumsum'] = insample_df['Manual Orders'].cumsum()
    previous_cumsum = None

    for date, cumsum_value in insample_df['Manual Orders Cumsum'].items():
        # Cumsum turns pos
        if cumsum_value > 0 and (previous_cumsum is None or previous_cumsum <= 0):
            ax1.axvline(date, color='blue', linestyle='--', linewidth=1)
        # Cumsum turns neg
        elif cumsum_value < 0 and (previous_cumsum is None or previous_cumsum >= 0):

            ax1.axvline(date, color='black', linestyle='--', linewidth=1)


        # Update the previous cumsum value
        previous_cumsum = cumsum_value

    custom_legend = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Long'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Short'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=1, label='Benchmark Portfolio Value'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Manual Strategy Portfolio Value')
    ]
    # Display legends and title
    ax1.legend(loc='upper left', handles=custom_legend)
    plt.title('In-Sample Manual vs Benchmark Strategy')
    #plt.show()
    plt.savefig('insample_strategy_orders.png')




    #calculate returns
    metric_df = pd.DataFrame(columns=['Strategy','Cumulative Return', 'Standard Deviation Daily Returns', 'Average Daily Returns'])
    #metric_df['Strategy'] = ['Optimal Portfolio','Benchmark Portfolio']
    manual_metrics = strategy.calculate_metrics(manual_strategy_port_val['Manual Strategy Portfolio Value'])
    manual_metrics.insert(0,'Manual Strategy Portfolio')
    benchmark_metrics = strategy.calculate_metrics(benchmark_port_val['Benchmark Portfolio Value'])
    benchmark_metrics.insert(0,'Benchmark Portfolio')
    metric_df.loc[len(metric_df)] = manual_metrics
    metric_df.loc[len(metric_df)] = benchmark_metrics


    metric_df.to_csv('In Sample Strategy_Metric_Table.csv', index=False)











    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    outsample_df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]
    outsample_df.rename(columns={symbol: 'Adj Close'}, inplace=True)

    # out sample manual Strategy
    manual_orders = strategy.testPolicy(sd=sd, ed=ed)
    manual_orders['Symbol'] = symbol
    manual_strategy_port_val = compute_portvals(df=manual_orders)
    manual_orders['Manual Orders'] = manual_orders['Order']
    manual_strategy_port_val.columns = ['Manual Strategy Portfolio Value']

    # out sample Benchmark Strategy
    benchmark_orders = manual_orders
    benchmark_orders['Order'] = 0
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.iloc[0, 0] = 1000
    benchmark_port_val = compute_portvals(df=benchmark_orders, start_val=100000, impact=0.005, commission=9.95)
    benchmark_port_val.columns = ['Benchmark Portfolio Value']

    outsample_df = pd.concat(
        [outsample_df, benchmark_port_val, manual_strategy_port_val, manual_orders[['Manual Orders']]], axis=1)

    # out sample Strategy plot
    plt.figure(figsize=(9, 6))
    outsample_df[['Normalized Manual Strategy Portfolio Value']] = outsample_df[['Manual Strategy Portfolio Value']].div(
        outsample_df['Manual Strategy Portfolio Value'].iloc[0])
    outsample_df[['Normalized Benchmark Portfolio Value']] = outsample_df[['Benchmark Portfolio Value']].div(
        outsample_df['Benchmark Portfolio Value'].iloc[0])

    ax1 = plt.gca()

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax1.set_xlim([outsample_df.index.min(), outsample_df.index.max()])
    plt.xticks(rotation=90)

    ax1.plot(outsample_df.index, outsample_df['Normalized Manual Strategy Portfolio Value'], color='red',
             label='Normalized Manual Strategy Portfolio Value')
    ax1.plot(outsample_df.index, outsample_df['Normalized Benchmark Portfolio Value'], color='purple',
             label='Normalized Benchmark Portfolio Value')
    ax1.set_ylabel('Normalized Portfolio Value')
    ax1.tick_params(axis='y')

    # Loop through dates and cumulative sum values to plot lines based on conditions
    outsample_df['Manual Orders Cumsum'] = outsample_df['Manual Orders'].cumsum()
    previous_cumsum = None

    for date, cumsum_value in outsample_df['Manual Orders Cumsum'].items():
        # Cumsum turns pos
        if cumsum_value > 0 and (previous_cumsum is None or previous_cumsum <= 0):
            ax1.axvline(date, color='blue', linestyle='--', linewidth=1)
        # Cumsum turns neg
        elif cumsum_value < 0 and (previous_cumsum is None or previous_cumsum >= 0):

            ax1.axvline(date, color='black', linestyle='--', linewidth=1)

        # Update the previous cumsum value
        previous_cumsum = cumsum_value

    custom_legend = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Long'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Short'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=1, label='Benchmark Portfolio Value'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Manual Strategy Portfolio Value')
    ]
    # Display legends and title
    ax1.legend(loc='upper left', handles=custom_legend)
    plt.title('Out-Sample Manual vs Benchmark Strategy')
    #plt.show()
    plt.savefig('out_sample_strategy_orders.png')


    #calculate returns
    metric_df = pd.DataFrame(columns=['Strategy','Cumulative Return', 'Standard Deviation Daily Returns', 'Average Daily Returns'])
    #metric_df['Strategy'] = ['Optimal Portfolio','Benchmark Portfolio']
    manual_metrics = strategy.calculate_metrics(manual_strategy_port_val['Manual Strategy Portfolio Value'])
    manual_metrics.insert(0,'Manual Strategy Portfolio')
    benchmark_metrics = strategy.calculate_metrics(benchmark_port_val['Benchmark Portfolio Value'])
    benchmark_metrics.insert(0,'Benchmark Portfolio')
    metric_df.loc[len(metric_df)] = manual_metrics
    metric_df.loc[len(metric_df)] = benchmark_metrics


    metric_df.to_csv('Out Sample Strategy_Metric_Table.csv', index=False)


