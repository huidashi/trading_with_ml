from util import get_data
import marketsimcode as ms
#import TheoreticallyOptimalStrategy as ts
import datetime as dt
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def author():
  return 'hshi320'



def calculate_momentum(price_series):
  previous_price = price_series.shift(1)
  momentum = ((price_series / previous_price)-1)*100
  return momentum

def calculate_SMA20(price_series):
  sma = price_series.rolling(20).mean()
  return sma

#
def calculate_rsi(price_series):
  prev_price = price_series - price_series.shift(1)
  total_gain = prev_price.where(prev_price > 0, 0)
  total_loss = -1*prev_price.where(prev_price < 0, 0)

  avg_gain = total_gain.rolling(20).mean()
  avg_loss = total_loss.rolling(20).mean()


  rsi = 100 - (100/ (1 + (avg_gain/avg_loss)))

  #fix rsi when avg_loss = 0 because cant divide by 0
  fifty_fifty = (avg_loss==0) & (avg_gain==0)
  zero_loss_boolean = (avg_loss==0)

  rsi.loc[zero_loss_boolean] = 100
  rsi.loc[fifty_fifty] = 50
  return rsi
#

def calculate_bollinger_bands_percent(price_series):
  sma = calculate_SMA20(price_series)
  rolling_std = price_series.rolling(20).std()

  lower_band = sma - (rolling_std * 2) #2stdev
  upper_band = sma + (rolling_std * 2)
  percent = (price_series - lower_band) / (upper_band - lower_band) * 100
  return percent

def calculate_stochastic_oscillator(price_series, high_series, low_series):
  lowest = low_series.rolling(20).min()
  highest = high_series.rolling(20).max()
  k_percent = (price_series-lowest) / (highest - lowest) * 100
  return k_percent

def calculate_commodity_channel_index(price_series, high_series, low_series):
  avg_prices = (high_series+low_series+price_series) / 3
  sma_avg_prices = avg_prices.rolling(20).mean()
  abs_diff = abs(avg_prices - sma_avg_prices)
  mean_deviation = abs_diff.rolling(20).mean()
  cci = (avg_prices - sma_avg_prices) / (0.015 * mean_deviation)

  return cci



def plot_indicators(symbol='JPM'):
  #symbol = 'JPM'
  sd = dt.datetime(2008, 1, 1)
  ed = dt.datetime(2009, 12, 31)
  df = get_data([symbol], pd.date_range(sd, ed), colname='Adj Close')[[symbol]]#.rename(columns={'JPM':'Adj Close'})
  df.rename(columns={'JPM': 'Adj Close'}, inplace=True)

  #adjusted high and lows
  df['Close'] = get_data([symbol], pd.date_range(sd, ed), colname='Close')[[symbol]]
  adjusted_ratio = df['Adj Close']/df['Close']

  df['High'] = get_data([symbol], pd.date_range(sd, ed), colname='High')[[symbol]]
  df['High'] = df['High'] * adjusted_ratio
  df['Low'] = get_data([symbol], pd.date_range(sd, ed), colname='Low')[[symbol]]
  df['Low'] = df['Low'] * adjusted_ratio

  #indicators
  df['Commodity Channel Index'] = calculate_commodity_channel_index(price_series=df['Adj Close'], high_series=df['High'], low_series=df['Low'])
  df['Bollinger Bands Percent'] = calculate_bollinger_bands_percent(df['Adj Close'])
  df['Stochastic Oscillator'] = calculate_stochastic_oscillator(df['Adj Close'], high_series=df['High'], low_series=df['Low'])
  df['SMA20'] = calculate_SMA20(df['Adj Close'])
  df['Momentum'] = calculate_momentum(df['Adj Close'])

  #stochastic
  plt.figure(figsize=(10, 6))

  ax1 = plt.gca()
  adjusted_close = ax1.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Adj Close $')
  ax1.tick_params(axis='y')
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
  plt.xticks(ticks=df.index[::30], rotation=90)

  ax2 = ax1.twinx()
  stochastic_line = ax2.plot(df.index, df['Commodity Channel Index'], color='lightsteelblue', label='Stochastic Oscillator K%')
  ax2.set_ylabel('Stochastic Oscillator')
  ax2.tick_params(axis='y')

  ax1.legend(loc = 'upper left')
  ax2.legend(loc = 'upper right')
  plt.title('Stochastic Oscillator')
  #plt.show()
  plt.savefig('Stochastic_Oscillator.png')


  #sma20
  plt.figure(figsize=(10, 6))

  ax1 = plt.gca()
  adjusted_close = ax1.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Adj Close $')
  ax1.tick_params(axis='y')
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
  plt.xticks(ticks=df.index[::30], rotation=90)

  SMA20 = ax1.plot(df.index, df['SMA20'], color='rosybrown', label='20 Day SMA')

  ax1.legend()
  ax2.legend()
  plt.title('20 Day SMA')
  #plt.show()
  plt.savefig('SMA20.png')

  #momentum
  plt.figure(figsize=(10, 6))

  ax1 = plt.gca()
  adjusted_close = ax1.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Adj Close $')
  ax1.tick_params(axis='y')
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
  plt.xticks(ticks=df.index[::30], rotation=90)

  ax2 = ax1.twinx()
  momentum = ax2.plot(df.index, df['Momentum'], color='lightsteelblue',
                             label='Momentum %')
  ax2.set_ylabel('Momentum %')
  ax2.tick_params(axis='y')

  ax1.legend(loc = 'upper left')
  ax2.legend(loc = 'upper right')
  plt.title('Momentum')
  #plt.show()
  plt.savefig('Momentum.png')

  # CCI
  plt.figure(figsize=(10, 6))

  ax1 = plt.gca()
  adjusted_close = ax1.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Adj Close $')
  ax1.tick_params(axis='y')
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
  plt.xticks(ticks=df.index[::30], rotation=90)

  ax2 = ax1.twinx()
  momentum = ax2.plot(df.index, df['Commodity Channel Index'], color='lightsteelblue',
                      label='Commodity Channel Index')
  ax2.set_ylabel('Commodity Channel Index')
  ax2.tick_params(axis='y')

  ax1.legend(loc = 'upper left')
  ax2.legend(loc = 'upper right')
  plt.title('Commodity Channel Index')
  #plt.show()
  plt.savefig('Commodity_Channel_Index.png')

  # bb%
  plt.figure(figsize=(10, 6))

  ax1 = plt.gca()
  adjusted_close = ax1.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Adj Close $')
  ax1.tick_params(axis='y')
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
  plt.xticks(ticks=df.index[::30], rotation=90)

  ax2 = ax1.twinx()
  momentum = ax2.plot(df.index, df['Bollinger Bands Percent'], color='lightsteelblue',
                      label='Bollinger Bands Percent')
  ax2.set_ylabel('Bollinger Bands Percent')
  ax2.tick_params(axis='y')

  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')
  plt.title('Bollinger Bands Percent')
  #plt.show()
  plt.savefig('Bollinger_Bands_Percent.png')

if __name__ == "__main__":
  plot_indicators()
  #print(df['CCI'])