# -*- coding: utf-8 -*-
"""

@author: Adele Ravagnani 

This module contains a class which allows to perform the execution of a meta order with a naive trading strategy.
The trading is split in equally spaced (by a given trading interval) child MOs of unitary size and equal direction.
This class is employed by the function "simulate_LOB_and_NaiveTrading" in the "NMSF.py" module.

"""
import numpy as np

#%%

class NaiveTrading:
    def __init__(self, trading_interval, total_childMOs, direction):
        """
        This class allows to perform the execution of a meta order with a naive trading strategy.

        Parameters
        ----------
        trading_interval : int
            This is the trading interval i.e. the time step (in event time) between 2 consecutive child MOs.
        total_childMOs : int
            This is the number of the child MOs which constitue the meta order.
        direction : int, it can be +1 or -1
            If it is +1, we have a buy meta order and so, all child MOs are with a buy direction.
            If it is -1, we have a sell meta order and so, all child MOs are with a sell direction.

        Returns
        -------
        None.

        """
        
        super().__init__()

        self.shares_interval = 1 #SF: each order has unitary size
        self.trading_interval = trading_interval 
        self.total_childMOs = total_childMOs 
        self.direction = direction
        self.shares_to_execute = self.total_childMOs
        
        self.trades_list = [] #it contains the indices of the market events corresponding to child MOs
        
    def trade(self, LOB_simulation):
        """
        This function performs the trade of a child MO.

        Parameters
        ----------
        LOB_simulation : class
            This is the class defined in "NMSF.py"; it represents the LOB and stores its evolution.

        Raises
        ------
        ValueError
            An error is raised if the child MO leads to the emptying of the book.

        Returns
        -------
        price : int
            Index of the price at which the child MO is executed.
        shift_lob_state : int
            Shift related to the centering of the book after the LOB update due to the child MO.

        """
        
        if self.direction == -1 and LOB_simulation.lob_state[LOB_simulation.lob_state > 0].sum() == self.shares_interval:
            raise ValueError('There is not enough volume in the book to execute the shares of the strategy. Try to increase the book size or the trading_interval.')
        elif self.direction == +1 and np.abs(LOB_simulation.lob_state[LOB_simulation.lob_state < 0]).sum() == self.shares_interval:
           raise ValueError('There is not enough volume in the book to execute the shares of the strategy. Try to increase the book size or the trading_interval.')
        else:
        
            LOB_simulation.mid_price_to_store['Previous'] = LOB_simulation.compute_mid_price() 
            
            price = LOB_simulation.compute_market_order_price(self.direction)
            LOB_simulation.lob_state[price] += self.direction*self.shares_interval
            LOB_simulation.remove_order_from_queue(price, 0)
            
            LOB_simulation.mid_price_to_store['Current'] = LOB_simulation.compute_mid_price() 
            LOB_simulation.exp_weighted_return_to_store = LOB_simulation.gamma_exp_weighted_return*LOB_simulation.exp_weighted_return_to_store + (LOB_simulation.mid_price_to_store['Current'] - LOB_simulation.mid_price_to_store['Previous'])
            
            shift_lob_state = LOB_simulation.center_lob_state()
            
        return price, shift_lob_state
       
