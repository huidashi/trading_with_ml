from util import get_data
import pandas as pd
import datetime as dt


def author():
  return 'hshi320'


def testPolicy(
  sd= dt.datetime(2010, 1, 1),
  ed= dt.datetime(2011,12,31),
  symbol = 'AAPL',
  sv = 100000
):
  df = get_data([symbol], pd.date_range(sd, ed))[[symbol]]

  orders = pd.DataFrame(index=df.index, columns=['Order'])
  orders['Order'] = 0
  holdings = 0

  for i in range(len(df)-1):
    # price go up next day max long
    if df.iloc[i+1][symbol] >df.iloc[i][symbol]:
      #check holdings
      if holdings == 0:
        orders.iloc[i] = 1000
        holdings = 1000
      elif holdings == -1000:
        orders.iloc[i] = 2000
        holdings = 1000
    elif df.iloc[i+1][symbol] < df.iloc[i][symbol]:
      if holdings ==0:
        orders.iloc[i] = -1000
      elif holdings == 1000:
        orders.iloc[i] = -2000
        holdings = -1000
    #do nothing position agree with future
    else:
      orders.iloc[i] = 0

  return orders



if __name__ == "__main__":
  print(testPolicy(symbol='JPM'))
  pass

