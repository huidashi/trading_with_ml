import marketsimcode as ms
import TheoreticallyOptimalStrategy as ts
import datetime as dt
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import indicators as id
import numpy as np

def author():
  return 'hshi320'

def calculate_metrics(port_val):
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]

    adr = daily_rets.mean()
    sddr = daily_rets.std()

    sr = (adr - 0) / sddr
    sr = (sr * m.sqrt(252))

    cr = (port_val[-1] / port_val[0]) - 1

    return ["{:.6f}".format(cr), round(sddr,6), round(adr,6)]

if __name__ == '__main__':

    #plot Optimal vs Benchmark graph and table
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009,12,31)
    optimal_orders = ts.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
    optimal_orders['Symbol'] = 'JPM'
    optimal_port_val = ms.compute_portvals(df=optimal_orders,  start_val=100000, commission = 0, impact = 0)
    optimal_port_val.columns = ['Optimal Portfolio Value']


    benchmark_orders = optimal_orders
    benchmark_orders['Order'] = 0
    benchmark_orders.iloc[0,0] = 1000
    benchmark_port_val = ms.compute_portvals(df=benchmark_orders, start_val=100000, commission = 0, impact = 0)
    benchmark_port_val.columns = ['Benchmark Portfolio Value']

    df_port_val = optimal_port_val.merge(benchmark_port_val, left_index=True, right_index=True)
    df_port_val[['Normalized Optimal Portfolio Value','Normalized Benchmark Portfolio Value']] = df_port_val[['Optimal Portfolio Value', 'Benchmark Portfolio Value']].div(df_port_val.iloc[0])
    plt.plot(df_port_val.index, df_port_val['Normalized Optimal Portfolio Value'], color = 'red', label='Optimal Portfolio Value')
    plt.plot(df_port_val.index, df_port_val['Normalized Benchmark Portfolio Value'], color = 'purple', label='Benchmark Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    plt.xticks(rotation=45)

    plt.title('Optimal vs Benchmark Strategy')
    plt.legend()
    #plt.show()
    plt.savefig('Optimal_vs_Benchmark_Strategy')

    metric_df = pd.DataFrame(columns=['Strategy','Cumulative Return', 'Standard Deviation Daily Returns', 'Average Daily Returns'])
    #metric_df['Strategy'] = ['Optimal Portfolio','Benchmark Portfolio']

    optimal_metrics = calculate_metrics(optimal_port_val['Optimal Portfolio Value'])

    optimal_metrics.insert(0,'Optimal Portfolio')
    benchmark_metrics = calculate_metrics(benchmark_port_val['Benchmark Portfolio Value'])
    benchmark_metrics.insert(0,'Benchmark Portfolio')
    metric_df.loc[len(metric_df)] = optimal_metrics
    metric_df.loc[len(metric_df)] = benchmark_metrics


    metric_df.to_csv('Strategy_Metric_Table.csv', index=False)



    #plot and save indicators graphs
    id.plot_indicators()

