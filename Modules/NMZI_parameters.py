# -*- coding: utf-8 -*-
"""

@author: Adele Ravagnani

In this module there is a function which allows to estimate the parameters of the Zero Intelligence model (aka Santa Fe model).

Reference to the Chapter 8 of the book: Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). Trades, quotes and prices: Financial markets under the microscope. Cambridge University Press.

"""

import numpy as np

#%%
def estimate_parameters_ZI(LOB_data, verbose = True):
    """
    Function which allows to estimate the parameters of the Zero Intelligence model from empirical data.

    Parameters
    ----------
    LOB_data : class
        "LOB_data" class where the empirical data set needs to be loaded.
        We are going to use the "message_file" and the "ob_file" instances.
    verbose : bool, optional
        The default is True; in this case, some information about the results are printed.

    Returns
    -------
    v_0 : float
        It is the mean size of limit orders (LOs). 
        Only LOs placed at the best quotes or inside the spread are considered.
    lam : float
        It is the total LO arrival rate per order per unit price.
    mu : float
        It is the total market order (MO) arrival rate per event.
    delta : float
        It is the total cancellation rate per unit volume and per event.
        Only cancellations that occur at the best quotes are considered.
    mean_inter_arrival_times: float
        It is the mean inter-arrival time between orders.
        
    """
    
    message_file = LOB_data.message_file
    ob_file = LOB_data.ob_file
    tick_size = LOB_data.tick_size
    
    """
    Remember that ob_file.iloc[j] is the state of the book after the event message_file.iloc[j]
    """
    
    #lo = limit order, c = cancellation, mo = market order
    
    #consider only LOs placed at the best quotes or inside the spread 
    #i.e. orders with price p_t \in [b_{t-1}, a_{t-1}]
    ind_lo = np.where(message_file['Type'] == 1)[0]
    
    ind_lo_close_to_midprice = []
    for ind in ind_lo:
        if ind > 0:
            price_lo = message_file['Price'].iloc[ind]
            best_ask_before_lo = ob_file['AskPrice_1'].iloc[ind - 1]
            best_bid_before_lo = ob_file['BidPrice_1'].iloc[ind - 1]
            if best_bid_before_lo <= price_lo <= best_ask_before_lo:
                ind_lo_close_to_midprice.append(ind)
    
    message_file_lo = message_file.iloc[ind_lo_close_to_midprice]  
    ob_file_before_lo = ob_file.iloc[[ind - 1 for ind in ind_lo_close_to_midprice]] #by def each element in ind_lo_close_midprice is > 0
    spread_before_lo_tick_size = ob_file_before_lo['AskPrice_1']/tick_size - ob_file_before_lo['BidPrice_1']/tick_size
    
    #consider only cancellations that occur at the best quotes
    ind_c = np.where((message_file['Type'] == 2) | (message_file['Type'] == 3))[0]
    
    ind_c_best_quotes = []
    for ind in ind_c:
        if ind > 0:
            price_c = message_file['Price'].iloc[ind]
            best_ask_before_c = ob_file['AskPrice_1'].iloc[ind - 1]
            best_bid_before_c = ob_file['BidPrice_1'].iloc[ind - 1]
            if price_c == best_ask_before_c or price_c == best_bid_before_c:
                ind_c_best_quotes.append(ind)
    
    message_file_c = message_file.iloc[ind_c_best_quotes]  
    
    #by def, MOs occur at the best quotes (hidden orders are excluded)
    message_file_mo = message_file[(message_file['Type'] == 4)]
    
    N_lo = len(message_file_lo)
    N_c = len(message_file_c)
    N_mo = len(message_file_mo)
    N_tot = N_lo + N_c + N_mo
    
    volumes_lo = message_file_lo['Size']
    volumes_c = message_file_c['Size']
    volumes_mo = message_file_mo['Size']
    
    #mean size of LOs
    v_0 = volumes_lo.mean()
    
    #total MO arrival rate per event
    mu = volumes_mo.sum()/(N_tot*2*v_0)
    
    #n = mean number of available price levels inside the spread and at the best quotes, measured only at the times of limit order arrivals
    n = 2*(1 + (np.floor(spread_before_lo_tick_size/2)).mean())
    
    #total LO arrival rate per event per unit price
    lam_all = N_lo/(2*N_tot)
    lam = lam_all/n
    
    #total cancellation rate per unit volume and per event
    V_bar = 0.5*(ob_file['AskSize_1'].mean() + ob_file['BidSize_1'].mean())
    delta = volumes_c.sum()/(2*N_tot*V_bar)
    
    #lambda interval arrival times
    mean_inter_arrival_times = message_file['Time'].diff(+1).mean()
    
    if verbose == True:
        print('Parameters estimation')
        print('The mean size of LOs (in number of shares) is %.4f.'%v_0)
        print('The total LO arrival rate per event per unit price is %.4f.'%lam)
        print('The total MO arrival rate per event is %.4f.'%mu)
        print('The total cancellation rate per unit volume and per event is %.4f.'%delta)
        print('The mean inter-arrival time between orders is %.4f seconds.'%mean_inter_arrival_times)
    
    return v_0, lam, mu, delta, mean_inter_arrival_times
