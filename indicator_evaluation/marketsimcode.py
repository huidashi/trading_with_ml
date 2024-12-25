import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return 'hshi320'  # replace tb34 with your Georgia Tech username.


def row_port_value(row, holdings, symbols_list, cash_balance, impact, commission):
    # did not buy or sell for day
    if pd.notna(row['Symbol']) and pd.notna(row['Order']):
        stock_price = row[row['Symbol']]
        # buy stock
        if row['Order'] > 0:
            adjusted_price_impact = stock_price * (1 + impact)
            total_cost = adjusted_price_impact * abs(row['Order']) + commission
            cash_balance = cash_balance - total_cost
            holdings[row['Symbol']] = holdings[row['Symbol']] + abs(row['Order'])
        # sell stock
        elif row['Order'] < 0:
            adjusted_price_impact = stock_price * (1 - impact)
            total_revenue = adjusted_price_impact * abs(row['Order']) - commission
            cash_balance = cash_balance + total_revenue
            holdings[row['Symbol']] = holdings[row['Symbol']] - abs(row['Order'])
    portfolio_row_value = sum(holdings[stock] * row[stock] for stock in symbols_list if holdings[stock] > 0)
    portfolio_row_short_value = sum(abs(holdings[stock] * row[stock]) for stock in symbols_list if holdings[stock] < 0)

    total_row_value = portfolio_row_value - portfolio_row_short_value + cash_balance
    return total_row_value, cash_balance


def compute_portvals(
        start_val=1000000,
        commission=9.95,
        impact=0.005,
        df = pd.DataFrame()
):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 	   		  		  		    	 		 		   		 		  

    :param orders_file: Path of the order file or the file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		 	   		  		  		    	 		 		   		 		  
    # TODO: Your code here  		  	   		 	   		  		  		    	 		 		   		 		  

    # In the template, instead of computing the value of the portfolio, we just  		  	   		 	   		  		  		    	 		 		   		 		  
    # read in the value of IBM over 6 months

    # read
    orders_df = df
    # dates
    start_date = orders_df.index.values[0]
    end_date = orders_df.index.values[-1]

    # unique symbols from orders
    symbols_list = orders_df['Symbol'].unique().tolist()

    # daily stock price date explode left join on orders
    stock_matrix = get_data(symbols_list, pd.date_range(start_date, end_date))
    port_matrix = stock_matrix.merge(orders_df, how='left', left_index=True, right_index=True)

    holdings = dict.fromkeys(symbols_list, 0)
    cash_balance = start_val

    port_values = []
    cash_balances = []

    # loop everyday, update portfolio holdings and calculate portfolio value
    for index, row in port_matrix.iterrows():
        port_value, cash_balance = row_port_value(row, holdings, symbols_list, cash_balance, impact, commission)
        port_values.append(port_value)
        cash_balances.append(cash_balance)

    port_matrix['Portfolio Value'] = port_values

    final_df = port_matrix.groupby(port_matrix.index).last([['Portfolio Value']])

    final_df = final_df[['Portfolio Value']]  # remove SPY
    rv = pd.DataFrame(index=final_df.index, data=final_df.values)
    return rv


# def test_code():
#     """
#     Helper function to test code
#     """
#     # this is a helper function you can use to test your code
#     # note that during autograding his function will not be called.
#     # Define input parameters
#
#     of = "./orders/orders-02.csv"
#     sv = 1000000
#
#     # Process orders
#     portvals = compute_portvals(orders_file=of, start_val=sv)
#     print(portvals)
#     if isinstance(portvals, pd.DataFrame):
#         portvals = portvals[portvals.columns[0]]  # just get the first column
#     else:
#         "warning, code did not return a DataFrame"
#
#     # Get portfolio stats
#     # Here we just fake the data. you should use your code from previous assignments.
#     start_date = dt.datetime(2008, 1, 1)
#     end_date = dt.datetime(2008, 6, 1)
#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
#         0.2,
#         0.01,
#         0.02,
#         1.5,
#     ]
#     cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
#         0.2,
#         0.01,
#         0.02,
#         1.5,
#     ]
#
#     # Compare portfolio against $SPX
#     print(f"Date Range: {start_date} to {end_date}")
#     print()
#     print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
#     print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
#     print()
#     print(f"Cumulative Return of Fund: {cum_ret}")
#     print(f"Cumulative Return of SPY : {cum_ret_SPY}")
#     print()
#     print(f"Standard Deviation of Fund: {std_daily_ret}")
#     print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
#     print()
#     print(f"Average Daily Return of Fund: {avg_daily_ret}")
#     print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
#     print()
#     print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    pass
    #test_code()