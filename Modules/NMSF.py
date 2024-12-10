# -*- coding: utf-8 -*-
"""

@author: Adele Ravagnani

This module contains the implementation of the Non-Markovian Zero Intelligence model.
Simulations can be performed by means of the functions:
    1) "simulate_LOB", without any execution of meta orders;
    2) "simulate_LOB_and_NaiveTrading" which allows to interact with the simulator and execute a meta order.
These two functions employ the class "LOB_simulation" that represents the LOB and stores its updates during the simulation.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

import sys
sys.path.append('/adeleravagnani/Non-narkovian-santa-fe-lob-simulator/Modules/')
import NMSF_trading as NMSF_t

#%%

class LOB_simulation:
    
    def __init__(self, number_tick_levels, n_priority_ranks, p0, v0, 
                 number_levels_to_store, beta_exp_weighted_return, 
                 intensity_exp_weighted_return):
        """
        First, consider that in this class, the LOB is represented by an array ("lob_state") where each entry stores the volume associated to a given price.
        Negative volumes are associated to the ask side and positive to the bid side.

        Parameters
        ----------
        number_tick_levels : int
            This is the length of the grid which represents the LOB i.e. the number of tick levels we consider.
            A suitable choice is a system of size at least ten times larger than the average spread, which allows the system to equilibrate relatively quickly.
        n_priority_ranks : int
            In order to represent the LOB, we also store "priorities_lob_state" that is a matrix of dimension n_priority_ranks X number_tick_levels.
            Each column of this matrix contains the volumes of the orders related to a given queue.
            These orders are ordered on a time-priority basis.
            So, n_priority_ranks represents the number of orders we want to store for each queue.
        p0 : int
            This is the initial price we associate to the first tick level in our grid.
        v0 : int
            This the unitary size of the orders.
        number_levels_to_store : int
            This is the number of the price levels we want to store in the output files ("ob_dict" and "message_dict").
            E.g. number_levels_to_store = 20: we store the first 10 levels of the ask and the first 10 levels of the bid side.
        beta_exp_weighted_return : float
            This is the beta parameter entering the definition of the exponentially weighted mid-price return ("exp_weighted_return_to_store").
        intensity_exp_weighted_return : float
            This is the intensity parameter (alpha) which enters the definition of the probability of a sell LO at time t given a LO at time t.
            If intensity_exp_weighted_return = 0, we recover the standard Zero Intelligence Santa Fe model.

        Returns
        -------
        None.

        """
        
        super().__init__()
        
        self.lob_state = None
        self.number_tick_levels = number_tick_levels 
        
        self.priorities_lob_state = None
        self.n_priority_ranks = n_priority_ranks
        
        self.p0 = p0
        self.v0 = v0
        
        self.number_levels_to_store = number_levels_to_store
        self.ob_dict = dict()
        self.message_dict = dict()
        
        self.message_df_simulation = None
        self.ob_df_simulation = None
        
        #Parameters of the Non-Markovian Zero Intelligence model
        self.beta_exp_weighted_return = beta_exp_weighted_return 
        self.gamma_exp_weighted_return = np.exp(-self.beta_exp_weighted_return)
        self.intensity_exp_weighted_return = intensity_exp_weighted_return
        
        self.mid_price_to_store = {'Current': 0, 'Previous': 0}
        self.exp_weighted_return_to_store = 0
        
        #The following lists store LOs submitted by the user and MOs which hit LOs by the user
        self.LO_by_user = [] #[priority_rank, price]
        self.passiveMOs_by_user = [] #each element is [priority_rank, price]
        
    def initialize_message_and_ob_dictionary(self):
        """
        This function initializes: 
            1) "ob_dict" that is dictionary where we store the prices and volumes for the first "number_levels_to_store" levels;
            2) "message_dict" that is a dictionary which stores the detials of each event.
        These two dictionaries are analogous to the message and order book file of the LOBSTER data.
        
        Returns
        -------
        None.

        """
    
        # initialize order book dictionary
        keys_ob_dict = []
        for i in range(1, self.number_levels_to_store//2 + 1):
            keys_ob_dict.append("AskPrice_" + str(i))
            keys_ob_dict.append("AskSize_" + str(i))
            keys_ob_dict.append("BidPrice_" + str(i))
            keys_ob_dict.append("BidSize_" + str(i))
    
        for key in keys_ob_dict:
            self.ob_dict[key] = []
    
        #initialize message dictionary
        keys_message_dict = ['Time', 'Type', 'Price', 'Direction', 'Spread', 'MidPrice', 'Shift', 'Return', 'TotNumberBidOrders', 'TotNumberAskOrders',
                             'IndBestBid', 'IndBestAsk'] 
        #in message_dict, rows i contains the type, price and direction of event i
        #it also contains the shift, spread, midprice and other info of the ob state after event i 
        for key in keys_message_dict:
            self.message_dict[key] = []
        
        return
    
    def initialize_lob_state(self):
        """
        This function initializes "lob_state" that is an array where each entry stores the volume of the orders associated to a given price.
        Negative volumes are associated to the ask side and positive to the bid side.
        
        By referring to Bouchaud et al.. 2018, "we choose the initial state of the LOB such that each price level is occupied by exactly one limit order. 
        Because small-tick LOBs exhibit gaps between occupied price levels and because each price level in large-tick LOBs is typically occupied by multiple limit orders, this initial condition in fact corresponds to a rare out-of-equilibrium state whose evolution allows one to track the equilibration process."        

        Returns
        -------
        None.
        
        """
        self.lob_state = np.zeros(self.number_tick_levels)
        self.lob_state[:self.number_tick_levels//2] = 1
        self.lob_state[self.number_tick_levels//2:] = -1
    
        return
    
    def initialize_priorities_lob_state(self):
        """
        This function initializes "priorities_lob_state" that is a matrix of dimension n_priority_ranks X number_tick_levels.
        Each column of this matrix contains the volumes of the orders related to a given queue.
        These orders are ordered on a time-priority basis.
        E.g. element (0, 0) stores the volume of the order with top priority at the deeper price level in the bid side of the grid which represents our LOB.
        We observe that storing these information is not necessary for simulating the Non-Markovian Zero Intelligence model. However, it can be useful if extensions of it are devised or different trading strategies are tested.

        Returns
        -------
        None.

        """
        self.priorities_lob_state = np.zeros((self.n_priority_ranks, self.number_tick_levels))
        self.priorities_lob_state[0, :self.number_tick_levels//2] = 1
        self.priorities_lob_state[0, self.number_tick_levels//2:] = -1
        
        return
    
    def compute_mid_price(self):
        """
        This function compute the mid-price of "lob_state".

        Returns
        -------
        float
            The position in "lob_state" which is associated to the mid-price is returned.
            It can assume half values.

        """
        best_bid = np.where(self.lob_state > 0)[0][-1]
        best_ask = np.where(self.lob_state < 0)[0][0]

        return (best_bid + best_ask)/2
    
    def compute_spread(self):
        """
        This function compute the spread in the book.

        Returns
        -------
        int
            It represents the spread in tick size.

        """
        best_bid = np.where(self.lob_state > 0)[0][-1]
        best_ask = np.where(self.lob_state < 0)[0][0]

        return best_ask - best_bid
    
    
    def initialize(self):
        """
        This function performs all the initialization steps necessary for building the class.

        Returns
        -------
        None.

        """
        self.initialize_lob_state()
        self.initialize_priorities_lob_state()
        self.initialize_message_and_ob_dictionary()
        self.mid_price_to_store['Current'] = self.compute_mid_price()
        self.mid_price_to_store['Previous'] = self.compute_mid_price()
        
        return 
    
    def center_lob_state(self):
        """
        This function centers, if it is necessary, the grid "lob_state" around the mid-price.
        
        Returns
        -------
        shift : int
            It is the shift we have performed by centering the LOB.

        """
        
        new_mid_price = self.compute_mid_price()
        shift = int(new_mid_price + 0.5 - self.number_tick_levels//2)

        if shift > 0:
            self.lob_state[:-shift] = self.lob_state[shift:]
            self.lob_state[-shift:] = np.zeros(len(self.lob_state[-shift:]))
            
            self.priorities_lob_state[:,:-shift] = self.priorities_lob_state[:, shift:]
            self.priorities_lob_state[:,-shift:] = np.zeros((len(self.priorities_lob_state[:, -shift:]), 1))
        
        elif shift < 0:
            self.lob_state[-shift:] = self.lob_state[:shift]
            self.lob_state[:-shift] = np.zeros(len(self.lob_state[:-shift]))
            
            self.priorities_lob_state[:,-shift:] = self.priorities_lob_state[:, :shift]
            self.priorities_lob_state[:,:-shift] = np.zeros((len(self.priorities_lob_state[:, :-shift]), 1))
        
        return shift
            
    def update_ob_dict(self, i):
        """
        This function update "ob_dict" and "message_dict" which are the dictionaries that store the states of the book and the events.

        Parameters
        ----------
        i : int
            It is the index of the iteration to store.

        Returns
        -------
        None.

        """
        # Check the number of non empty levels
        n_quotes_bid = self.lob_state[self.lob_state > 0].shape[0]
        n_quotes_ask = self.lob_state[self.lob_state < 0].shape[0]
        
        # Update bid price and bid volume for the first "number_levels_to_store//2" level
        for n in range(min(self.number_levels_to_store//2, n_quotes_bid)):
            self.ob_dict[f'BidPrice_{n + 1}'].append(np.where(self.lob_state > 0)[0][-n - 1])
            self.ob_dict[f'BidSize_{n + 1}'].append((self.lob_state[self.lob_state > 0][-n - 1])*self.v0)
        
        for n in range(self.number_levels_to_store//2):
            if len(self.ob_dict[f'BidPrice_{n + 1}']) < i + 1:
                self.ob_dict[f'BidPrice_{n + 1}'].append(0)
                self.ob_dict[f'BidSize_{n + 1}'].append(0)
                
        # Update ask price and ask volume for the first "number_levels_to_store//2" level
        for n in range(min(self.number_levels_to_store//2, n_quotes_ask)):
            self.ob_dict[f'AskPrice_{n + 1}'].append(np.where(self.lob_state < 0)[0][n])
            self.ob_dict[f'AskSize_{n + 1}'].append((-self.lob_state[self.lob_state < 0][n])*self.v0)
        
        for n in range(self.number_levels_to_store//2):
            if len(self.ob_dict[f'AskPrice_{n + 1}']) < i + 1:
                self.ob_dict[f'AskPrice_{n + 1}'].append(0)
                self.ob_dict[f'AskSize_{n + 1}'].append(0)
        
        return

    #--------------------------------------------------------------------------
    
    def draw_next_order_type(self, lam, mu, delta):
        """
        This function draws the next order type: LO, MO or cancellation. 

        Parameters
        ----------
        lam : float
            The estimated total LO arrival rate per event per unit price.
        mu : float
            The estimated total MO arrival rate per event.
        delta : float
            The estimated total cancellation rate per unit volume and per event.

        Returns
        -------
        type_next_order : int
            If it is 0, the sampled order is a LO.
            If it is 1, the sampled order is a MO.
            If it is 2, the sampled order is a cancellation.

        """
        
        #total rate of arrival of LOs
        Lam = lam*self.number_tick_levels 
        
        #total rate of arrival of MOs
        Mu = 2*mu
        
        #total cancellation rate for orders in the the book
        n_orders = np.abs(self.lob_state).sum() #each order has unitary size
        Delta = delta*n_orders
        
        Lam_time = np.random.exponential(1/Lam)
        Mu_time = np.random.exponential(1/Mu)
        Delta_time = np.random.exponential(1/Delta)
        type_next_order = np.argmin([Lam_time, Mu_time, Delta_time])
        
        return type_next_order  
        
    def draw_next_order(self, lam, mu, delta):
        """
        This function draws the next order type and sign.

        Parameters
        ----------
        lam : float
            The estimated total LO arrival rate per event per unit price.
        mu : float
            The estimated total MO arrival rate per event.
        delta : float
            The estimated total cancellation rate per unit volume and per event.

        Returns
        -------
        type_next_order : int
            If it is 0, the sampled order is a LO.
            If it is 1, the sampled order is a MO.
            If it is 2, the sampled order is a cancellation.
        sign_next_order : int
            If it is -1, the sampled sign is sell.
            If it is 1, the sampled sign is buy.

        """

        n_orders_bid = (self.lob_state[self.lob_state > 0]).sum()
        n_orders_ask = (np.abs(self.lob_state[self.lob_state < 0])).sum()
        
        type_next_order = self.draw_next_order_type(lam, mu, delta)
        
        # Assign sign
        if type_next_order == 0:#LO
            probability_sell = 1/(1 + np.exp(-self.intensity_exp_weighted_return*self.exp_weighted_return_to_store))
            sign_next_order = np.random.choice([1, -1], p = [1 - probability_sell, probability_sell])
        elif type_next_order == 1: #MO
            sign_next_order = np.random.choice([1, -1], p = [0.5, 0.5])
        else:
            sign_next_order = np.random.choice([1, -1], 
                    p = [n_orders_bid/(n_orders_bid + n_orders_ask), 
                         n_orders_ask/(n_orders_bid + n_orders_ask)])
        
        return type_next_order, sign_next_order
    
    
    def sample_limit_order_price(self, order_sign):
        """
        This function samples the price level of a LO.

        Parameters
        ----------
        order_sign : int
            +1: buy LO, -1: sell LO.

        Returns
        -------
        price : int
            It is the index in "lob_state" of the sampled price for the LO.

        """
        best_bid = np.where(self.lob_state > 0)[0][-1]
        best_ask = np.where(self.lob_state < 0)[0][0]

        if order_sign == +1:
            price = np.random.randint(0, best_ask)
        else:
            price = np.random.randint(best_bid + 1, self.number_tick_levels)

        return price   
    
    def compute_market_order_price(self, order_sign):
        """
        This function computes the price of a MO.

        Parameters
        ----------
        order_sign : int
            +1: buy MO, -1: sell LO.

        Returns
        -------
        price : int
            It is the index in "lob_state" of the MO's price.

        """

        if order_sign == +1:
            price  = np.where(self.lob_state < 0)[0][0]
        else:
            price = np.where(self.lob_state > 0)[0][-1]

        return price

    def sample_cancellation_price(self, order_sign):
        """
        This function samples the price level of a cancellation.

        Parameters
        ----------
        order_sign : int
            +1: buy cancellation, -1: sell cancellation.

        Returns
        -------
        price : int
            It is the index in "lob_state" of the sampled price for the cancellation.

        """
        
        n_orders_bid = (self.lob_state[self.lob_state > 0]).sum()
        n_orders_ask = (np.abs(self.lob_state[self.lob_state < 0])).sum()

        if order_sign == +1:
            ind_order_to_cancel = np.random.randint(n_orders_bid)
        else:
            ind_order_to_cancel = np.random.randint(n_orders_bid, n_orders_ask + n_orders_bid)
        
        numeration_orders = np.abs(self.lob_state).cumsum() 

        price = np.where(numeration_orders > ind_order_to_cancel)[0][0]
        
        return price
    
    def sample_cancellation_priority_rank(self, order_price):
        """
        This function samples the order which is cancelled at a given price.

        Parameters
        ----------
        order_price : int
            Price of the queue for which an order is cancelled.

        Returns
        -------
        priority_rank : int
            Index of the position in the queue of the order which is cancelled.

        """
        ind_priority_ranks = np.where(self.priorities_lob_state[:, order_price] != 0)[0]
        priority_rank = np.random.choice(ind_priority_ranks)
        
        return priority_rank
        
       
    #--------------------------------------------------------------------------
    def add_order_to_queue(self, order_price, order_signed_size):
        """
        Function that adds a given LO in the corresponding queue in "priorities_lob_state".

        Parameters
        ----------
        order_price : int
            Price of the order which has to be added.
        order_signed_size : int
            It is the volume of the order multiplied by its sign.

        Returns
        -------
        priority_rank : int
            Index of the position in the queue of the order which is added.

        """

        if len(np.where(self.priorities_lob_state[:, order_price] == 0)[0]) != 0:
            priority_rank = np.where(self.priorities_lob_state[:, order_price] == 0)[0][0]
            self.priorities_lob_state[priority_rank, order_price] += order_signed_size
        else:
            priority_rank = None
            warnings.warn('Cannot add the order to a queue i.e. the size of the order is added to the total volume in the book for its price but the position of the order in the queue is not registered. Try to increase the number of priority ranks which are stored.')
    
        return priority_rank
    
    def remove_order_from_queue(self, order_price, priority_rank):
        """
        This function removes an order in the corresponding queue in "priorities_lob_state", because of a MO or a cancellation.

        Parameters
        ----------
        order_price : int
            Price of the order which has to be removed because of a MO or a cancellation.
        priority_rank : int
            Index of the position in the queue of the order which is removed.

        Returns
        -------
        None.

        """

        if priority_rank < self.n_priority_ranks:
            self.priorities_lob_state[priority_rank:-1, 
                    order_price] = self.priorities_lob_state[priority_rank + 1:, 
                                                           order_price]
            self.priorities_lob_state[-1, order_price] = 0
        else:
            warnings.warn('Cannot find order in the queues.')
        
        return 
    
    #--------------------------------------------------------------------------
    def simulate_order(self, lam, mu, delta):
        """
        This function simulates the sampling of an order.

        Parameters
        ----------
        lam : float
            The estimated total LO arrival rate per event per unit price.
        mu : float
            The estimated total MO arrival rate per event.
        delta : float
            The estimated total cancellation rate per unit volume and per event.


        Returns
        -------
        order_type : int
            0: LO, 1: MO, 2: cancellation.
        order_sign : int
            +1: buy, -1: sell.
        order_price : int
            Index of the price level of the order.
        shift_lob_state : int
            Shift related to the centering of the book after the LOB update due to the sampled order.

        """
        
        n_orders_bid = (self.lob_state[self.lob_state > 0]).sum()
        n_orders_ask = (np.abs(self.lob_state[self.lob_state < 0])).sum()
        
        #store the mid-price in the same reference frame (i.e. before centering the lob state)
        #this mid-price will be employed to compute the returns which have to be used online to sample the LOs' signs
        self.mid_price_to_store['Previous'] = self.compute_mid_price() 
        
        flag = False
        while flag == False:
            [order_type, order_sign] = self.draw_next_order(lam, mu, delta)
            
            if order_type == 0: #LO
                order_price = self.sample_limit_order_price(order_sign)
                priority_rank = None

            elif order_type == 1: #MO
                order_price = self.compute_market_order_price(order_sign)
                priority_rank = 0

            else:
                order_price = self.sample_cancellation_price(order_sign)
                priority_rank = self.sample_cancellation_priority_rank(order_price)
            
            
            # Check that the order does not lead to the emptying of the book
            if n_orders_bid == 1 and order_sign == +1 and order_type == 2:
                flag = False

            elif n_orders_bid == 1 and order_sign == -1 and order_type == 1:
                flag = False

            elif n_orders_ask == 1 and order_sign == -1 and order_type == 2:
                flag = False

            elif n_orders_ask == 1 and order_sign == +1 and order_type == 1:
                flag = False
            
            #do not cancel orders submitted by the user
            elif order_type == 2 and [priority_rank, order_price] == self.LO_by_user:
                flag = False
                
            else:
                flag = True

        
        if order_type == 0: #LO
            self.lob_state[order_price] += order_sign*1
            self.add_order_to_queue(order_price, order_sign*1)

        elif order_type == 1: #MO
            self.lob_state[order_price] += order_sign*1
            self.remove_order_from_queue(order_price, priority_rank)
            
            if [priority_rank, order_price] == self.LO_by_user:
                self.LO_by_user = [order_price]
                self.passiveMOs_by_user.append([priority_rank, order_price])
        else: #C
            self.lob_state[order_price] -= order_sign*1
            self.remove_order_from_queue(order_price, priority_rank)
        
        #store the mid-price in the same reference frame (i.e. before centering the lob state)
        #this mid-price will be employed to compute the returns which have to be used online to sample the LOs' signs
        self.mid_price_to_store['Current'] = self.compute_mid_price()
        
        self.exp_weighted_return_to_store = self.gamma_exp_weighted_return*self.exp_weighted_return_to_store + (self.mid_price_to_store['Current'] - self.mid_price_to_store['Previous'])
        
        shift_lob_state = self.center_lob_state()
            
        return order_type, order_sign, order_price, shift_lob_state
    
    #--------------------------------------------------------------------------
    
    def fix_zero_size(self):
        """
        This function sets equal to 0 the prices corresponding to empty queues in "ob_df_simulation".

        Returns
        -------
        None.

        """

        header_price = [f"AskPrice_{i + 1}" for i in range(self.number_levels_to_store//2)] + [f"BidPrice_{i + 1}" for i in range(self.number_levels_to_store//2)]
        header_vol = [f"AskSize_{i + 1}" for i in range(self.number_levels_to_store//2)] + [f"BidSize_{i + 1}" for i in range(self.number_levels_to_store//2)]
        for price, size in zip(header_price, header_vol):
            #find all levels with 0 size and set price to 0
            zero_size = self.ob_df_simulation[self.ob_df_simulation[size] == 0].index
            self.ob_df_simulation.loc[zero_size, price] = 0
        
        
    def save_results(self, path_save_files, label_simulation, i_cut):
        """
        Function which saves the results of the simulation in two files: "ob_df_simulation" and "message_df_simulation".
        They are built analogously to the order book and message file of the LOBSTER format.

        Parameters
        ----------
        path_save_files : str 
            It is the path of the folder where the files have to be saved.
        label_simulation : str 
            It is the label which enters the name of the files which are stored.
        i_cut : int
            Index of the last iteration to consider when the files are saved.

        Returns
        -------
        None.

        """
        self.message_df_simulation = pd.DataFrame(self.message_dict).iloc[:i_cut + 1, :]
        self.ob_df_simulation = pd.DataFrame(self.ob_dict).iloc[:i_cut + 1, :]
        
        self.message_df_simulation['Type'].replace([0, 1, 2, 3, 4, 5], 
                                ['LO', 'MO', 'C', 'LOUser', 'AggressiveMOUser', 'PassiveMOUser'], inplace = True)
        
        # Correct prices
        increment_prices = self.message_df_simulation["Shift"].cumsum() + self.p0
        self.message_df_simulation["MidPrice"] += increment_prices.to_numpy()
        self.message_df_simulation.loc[1:, 'Price'] += increment_prices.to_numpy()[:-1] #shifted because event's price calculated before the shift of the lob state
        
        columns_ob_df_prices = [column for column in self.ob_df_simulation.columns if 'Price' in column]
        for column in columns_ob_df_prices:
            self.ob_df_simulation[column] += increment_prices.to_numpy()
    
        self.message_df_simulation.drop("Shift", axis = 1, inplace = True)
        
        self.fix_zero_size()
        
        if type(path_save_files) == str and type(label_simulation) == str:
            self.message_df_simulation.to_csv(path_save_files + 'message_file_simulation_' + label_simulation + '.csv')
            self.ob_df_simulation.to_csv(path_save_files + 'ob_file_simulation_' + label_simulation + '.csv')
            
        return
    
#%%
        
def simulate_LOB(lam, mu, delta, mean_inter_arrival_times, number_tick_levels, n_priority_ranks, number_levels_to_store = 20,
                 p0 = 0, v0 = 1, iterations = 50_000, iterations_to_equilibrium = 10_000,
                 path_save_files = None, label_simulation = None, beta_exp_weighted_return = 1e-3, intensity_exp_weighted_return = 1e-3):
    
    """
    This function allows to simulate the LOB evolution with the Non-Markovian Zero Intelligence model.

    Parameters
    ----------
    lam : float
        The estimated total LO arrival rate per event per unit price.
    mu : float
        The estimated total MO arrival rate per event.
    delta : float
        The estimated total cancellation rate per unit volume and per event.
    mean_inter_arrival_times : float
        This is the inter arrival times between events. It is used to sample a time for each event.
    number_tick_levels : int
        This is the length of the grid which represents the LOB i.e. the number of tick levels we consider.
        A suitable choice is a system of size at least ten times larger than the average spread, which allows the system to equilibrate relatively quickly.
    n_priority_ranks : int
        It represents the number of orders we want to store for each queue.
    number_levels_to_store : int, optional
       This is the number of the price levels we want to store in the output files. The default is 20.
    p0 : int
        This is the initial price we associate to the first tick level in our grid. The default is 0.
    v0 : int
        This the unitary size of the orders. The default is 1.
    iterations : int, optional
        This is the number of iterations in our simulation. The default is 50_000.
    iterations_to_equilibrium : TYPE, optional
        This is the number of iterations to reach equilibrium (their outputs are not saved). The default is 10_000.
    path_save_files : str 
        It is the path of the folder where the files have to be saved.
    label_simulation : str 
        It is the label which enters the name of the files which are stored.
    beta_exp_weighted_return : float
        This is the beta parameter entering the definition of the exponentially weighted mid-price return ("exp_weighted_return_to_store").
        The default is 1e-3.
    intensity_exp_weighted_return : float
        This is the intensity parameter (alpha) which enters the definition of the probability of a sell LO at time t given a LO at time t.
        If intensity_exp_weighted_return = 0, we recover the standard Zero Intelligence Santa Fe model. 
        The default is 1e-3.

    Returns
    -------
    pd.DataFrame
        It is the message file data frame. Row j stores the details of the event j.
    pd.DataFrame
        It is the order book file data frame. Row j stores the state of the book (volumes and prices) after the LOB update due to the event j.
    exp_weighted_return_list : list
        It stores the exponentially weighted mid-price return for each iteration in the simulation.

    """

    print('Let us simulate the Non-Markovian Zero Intelligence model.')
    print('We initialize the LOB ...')
    
    LOB_sim = LOB_simulation(number_tick_levels, n_priority_ranks, p0, v0, number_levels_to_store, beta_exp_weighted_return, intensity_exp_weighted_return) 
    
    LOB_sim.initialize()
    
    print('Now, the LOB is updated until the equilibrium is reached ...')
    for i in tqdm(range(iterations_to_equilibrium)):
        [_, _, _, _] = LOB_sim.simulate_order(lam, mu, delta)
    
    print('Let us start the "real" simulation and save the updates...')
    t_i = 0
    exp_weighted_return_list = []
    for i in tqdm(range(iterations)):
        t_i += np.random.exponential(mean_inter_arrival_times)
        LOB_sim.message_dict['Time'].append(t_i)
        
        [order_type, order_direction, order_price, shift_lob_state] = LOB_sim.simulate_order(lam, mu, delta)
        LOB_sim.message_dict['Type'].append(order_type) 
        LOB_sim.message_dict['Direction'].append(order_direction)
        LOB_sim.message_dict['Price'].append(order_price)
        LOB_sim.message_dict['Shift'].append(shift_lob_state)
        
        LOB_sim.message_dict['Spread'].append(LOB_sim.compute_spread())
        LOB_sim.message_dict['MidPrice'].append(LOB_sim.compute_mid_price())
        
        LOB_sim.message_dict['Return'].append(LOB_sim.mid_price_to_store['Current'] - LOB_sim.mid_price_to_store['Previous'])
        
        LOB_sim.message_dict['TotNumberBidOrders'].append((LOB_sim.lob_state[LOB_sim.lob_state > 0]).sum())
        LOB_sim.message_dict['TotNumberAskOrders'].append((np.abs(LOB_sim.lob_state[LOB_sim.lob_state < 0])).sum())
        
        LOB_sim.message_dict['IndBestBid'].append(np.where(LOB_sim.lob_state > 0)[0][-1])
        LOB_sim.message_dict['IndBestAsk'].append(np.where(LOB_sim.lob_state < 0)[0][0])
        
        #the element i associated to each key of ob_dict refers to the ob state after event i in message_dict
        LOB_sim.update_ob_dict(i)
        
        exp_weighted_return_list.append(LOB_sim.exp_weighted_return_to_store)
    
    print('Let us save the message and order book files in Pandas dataframes and let us save the two dataframes in csv files ...')
    LOB_sim.save_results(path_save_files, label_simulation, i)
    
    print('The end!')
    
    return LOB_sim.message_df_simulation, LOB_sim.ob_df_simulation, exp_weighted_return_list

#%%

def simulate_LOB_and_NaiveTrading(lam, mu, delta, mean_inter_arrival_times, 
                                  number_tick_levels, n_priority_ranks, 
                                  parameters_trading_strategy, 
                                  flag_trading_event_time = True,
                                  number_levels_to_store = 20,
                                  p0 = 0,
                                  v0 = 1,
                                  iterations_before_trading = 1_000,
                                  iterations_after_trading = 1_000, 
                                  iterations_to_equilibrium = 10_000,
                                  path_save_files = None, label_simulation = None,
                                  beta_exp_weighted_return = 1e-3, 
                                  intensity_exp_weighted_return = 1e-3):
    """
    This function allows to interact with the Non-Markovian Zero Intelligence simulator.
    Indeed, the LOB evolution is simulated while a meta order is executed with a naive trading strategy. The trading is split in equally spaced (by a given trading interval) child MOs of unitary size and equal direction.

    Parameters
    ----------
    lam : float
        The estimated total LO arrival rate per event per unit price.
    mu : float
        The estimated total MO arrival rate per event.
    delta : float
        The estimated total cancellation rate per unit volume and per event.
    mean_inter_arrival_times : float
        This is the inter arrival times between events. It is used to sample a time for each event.
    number_tick_levels : int
        This is the length of the grid which represents the LOB i.e. the number of tick levels we consider.
        A suitable choice is a system of size at least ten times larger than the average spread, which allows the system to equilibrate relatively quickly.
    n_priority_ranks : int
        It represents the number of orders we want to store for each queue.
    parameters_trading_strategy : list
        List of 3 elements that are the parameters of the trading strategy: the trading interval, the total shares to trade, the sign of the trades.
    flag_trading_event_time : bool, optional
        It is True if the unit of measure of the trading interval of the strategy is event time.
        It is False if the unit of measure of the trading interval of the strategy is seconds. 
        The default is True.
    number_levels_to_store : int, optional
       This is the number of the price levels we want to store in the output files. The default is 20.
    p0 : int
        This is the initial price we associate to the first tick level in our grid. The default is 0.
    v0 : int
        This the unitary size of the orders. The default is 1.
    iterations_before_trading : int, optional
        This is the number of iterations before starting to trade. The default is 1_000.
    iterations_after_trading : int, optional
        This is the number of iterations after the trading is concluded. The default is 1_000.
    iterations_to_equilibrium : int, optional
        This is the number of iterations to reach equilibrium (their outputs are not saved). The default is 10_000.
    path_save_files : str 
        It is the path of the folder where the files have to be saved.
    label_simulation : str 
        It is the label which enters the name of the files which are stored.
    beta_exp_weighted_return : float
        This is the beta parameter entering the definition of the exponentially weighted mid-price return ("exp_weighted_return_to_store").
        The default is 1e-3.
    intensity_exp_weighted_return : float
        This is the intensity parameter (alpha) which enters the definition of the probability of a sell LO at time t given a LO at time t.
        If intensity_exp_weighted_return = 0, we recover the standard Zero Intelligence Santa Fe model. 
        The default is 1e-3.

    Returns
    -------
    pd.DataFrame
        It is the message file data frame. Row j stores the details of the event j.
    pd.DataFrame
        It is the order book file data frame. Row j stores the state of the book (volumes and prices) after the update due to event j.
    NaiveTrading_class : class
        Class which allows to perform the trading with a naive strategy. It is defined in the module "MSF_trading.py".
    i_stop_trading : int
        Number of the iteration which corresponds to the end of trading.
    exp_weighted_return_list : list
        It stores the exponentially weighted mid-price return for each iteration in the simulation.
        
    """

    [trading_interval, total_childMOs_trading, direction] = parameters_trading_strategy
    
    NaiveTrading_class = NMSF_t.NaiveTrading(trading_interval, total_childMOs_trading, direction)

    print('Let us simulate the Non-Markovian Zero Intelligence model with the execution of a meta order and a naive trading strategy.')
    print('We initialize the LOB ...')
    
    LOB_sim = LOB_simulation(number_tick_levels, n_priority_ranks, p0, v0, 
                             number_levels_to_store, beta_exp_weighted_return,
                             intensity_exp_weighted_return) 
    
    LOB_sim.initialize()
    
    print('Now, the LOB is updated until the equilibrium is reached ...')
    for i in tqdm(range(iterations_to_equilibrium)):
        [_, _, _, _] = LOB_sim.simulate_order(lam, mu, delta)
        LOB_sim.exp_weighted_return_to_store = 0
    
    t_i = 0
    print('Let us start saving the updates...')
    for i in tqdm(range(iterations_before_trading)):
        t_i += np.random.exponential(mean_inter_arrival_times)
        LOB_sim.message_dict['Time'].append(t_i)
        
        [order_type, order_direction, order_price, shift_lob_state] = LOB_sim.simulate_order(lam, mu, delta)
        LOB_sim.message_dict['Type'].append(order_type) 
        LOB_sim.message_dict['Direction'].append(order_direction)
        LOB_sim.message_dict['Price'].append(order_price)
        LOB_sim.message_dict['Shift'].append(shift_lob_state)
        
        LOB_sim.message_dict['Spread'].append(LOB_sim.compute_spread())
        LOB_sim.message_dict['MidPrice'].append(LOB_sim.compute_mid_price())
        
        LOB_sim.message_dict['Return'].append(LOB_sim.mid_price_to_store['Current'] - LOB_sim.mid_price_to_store['Previous'])
        
        LOB_sim.message_dict['TotNumberBidOrders'].append((LOB_sim.lob_state[LOB_sim.lob_state > 0]).sum())
        LOB_sim.message_dict['TotNumberAskOrders'].append((np.abs(LOB_sim.lob_state[LOB_sim.lob_state < 0])).sum())
        
        LOB_sim.message_dict['IndBestBid'].append(np.where(LOB_sim.lob_state > 0)[0][-1])
        LOB_sim.message_dict['IndBestAsk'].append(np.where(LOB_sim.lob_state < 0)[0][0])
        
        #the element i associated to each key of ob_dict refers to the ob state after event i in message_dict
        LOB_sim.update_ob_dict(i)
        
        LOB_sim.exp_weighted_return_to_store = 0
    
    exp_weighted_return_list = []
    
    print('Let us start trading and save the updates...')
    t_reference = t_i
    i = iterations_before_trading
    i_reference = i
    pbar = tqdm()
    while 0 < NaiveTrading_class.shares_to_execute <= NaiveTrading_class.total_childMOs:
        
        t_i += np.random.exponential(mean_inter_arrival_times)
        LOB_sim.message_dict['Time'].append(t_i)
        
        if flag_trading_event_time == False:
            condition_to_trade = (t_i - t_reference >= NaiveTrading_class.trading_interval)
        else:
            condition_to_trade = (i - i_reference == NaiveTrading_class.trading_interval)
        
        if condition_to_trade:
            t_reference = t_i
            i_reference = i + 1
            
            [price, shift_lob_state] = NaiveTrading_class.trade(LOB_sim)

            LOB_sim.message_dict['Type'].append(4) #MO by user
            LOB_sim.message_dict['Direction'].append(NaiveTrading_class.direction)
            LOB_sim.message_dict['Price'].append(price)
            LOB_sim.message_dict['Shift'].append(shift_lob_state)
            
            NaiveTrading_class.trades_list.append(i)
            NaiveTrading_class.shares_to_execute -= NaiveTrading_class.shares_interval
            
        else:
            [order_type, order_direction, order_price, shift_lob_state] = LOB_sim.simulate_order(lam, mu, delta)
            LOB_sim.message_dict['Type'].append(order_type) 
            LOB_sim.message_dict['Direction'].append(order_direction)
            LOB_sim.message_dict['Price'].append(order_price)
            LOB_sim.message_dict['Shift'].append(shift_lob_state)
            
        LOB_sim.message_dict['Spread'].append(LOB_sim.compute_spread())
        LOB_sim.message_dict['MidPrice'].append(LOB_sim.compute_mid_price())
        
        LOB_sim.message_dict['Return'].append(LOB_sim.mid_price_to_store['Current'] - LOB_sim.mid_price_to_store['Previous'])
        
        LOB_sim.message_dict['TotNumberBidOrders'].append((LOB_sim.lob_state[LOB_sim.lob_state > 0]).sum())
        LOB_sim.message_dict['TotNumberAskOrders'].append((np.abs(LOB_sim.lob_state[LOB_sim.lob_state < 0])).sum())
        
        LOB_sim.message_dict['IndBestBid'].append(np.where(LOB_sim.lob_state > 0)[0][-1])
        LOB_sim.message_dict['IndBestAsk'].append(np.where(LOB_sim.lob_state < 0)[0][0])
        
        #the element i associated to each key of ob_dict refers to the ob state after event i in message_dict
        LOB_sim.update_ob_dict(i)
        
        if i - iterations_before_trading < NaiveTrading_class.trading_interval:
          LOB_sim.exp_weighted_return_to_store = 0
        
        exp_weighted_return_list.append(LOB_sim.exp_weighted_return_to_store)
        
        pbar.update(i)
        i += 1
    
    pbar.close()
    print('The execution is over after %s seconds.'%t_i)
    print('The number of iterations performed until the end of the execution is %s.'%i)
    i_stop_trading = i
    
    print('Let us finish the simulation...')
    for i in tqdm(range(i_stop_trading, i_stop_trading + iterations_after_trading)):
        t_i += np.random.exponential(mean_inter_arrival_times)
        LOB_sim.message_dict['Time'].append(t_i)
        
        [order_type, order_direction, order_price, shift_lob_state] = LOB_sim.simulate_order(lam, mu, delta)
        LOB_sim.message_dict['Type'].append(order_type) 
        LOB_sim.message_dict['Direction'].append(order_direction)
        LOB_sim.message_dict['Price'].append(order_price)
        LOB_sim.message_dict['Shift'].append(shift_lob_state)
        
        LOB_sim.message_dict['Spread'].append(LOB_sim.compute_spread())
        LOB_sim.message_dict['MidPrice'].append(LOB_sim.compute_mid_price()) 
        LOB_sim.message_dict['Return'].append(LOB_sim.mid_price_to_store['Current'] - LOB_sim.mid_price_to_store['Previous'])
        
        LOB_sim.message_dict['TotNumberBidOrders'].append((LOB_sim.lob_state[LOB_sim.lob_state > 0]).sum())
        LOB_sim.message_dict['TotNumberAskOrders'].append((np.abs(LOB_sim.lob_state[LOB_sim.lob_state < 0])).sum())
        
        LOB_sim.message_dict['IndBestBid'].append(np.where(LOB_sim.lob_state > 0)[0][-1])
        LOB_sim.message_dict['IndBestAsk'].append(np.where(LOB_sim.lob_state < 0)[0][0])
    
        #the element i associated to each key of ob_dict refers to the ob state after event i in message_dict
        LOB_sim.update_ob_dict(i)
        exp_weighted_return_list.append(LOB_sim.exp_weighted_return_to_store)
    
    print('The simulation ends after a total of %s seconds.'%t_i)
    print('Let us save the message and order book files in Pandas dataframes and let us save the two dataframes in csv files ...')
    LOB_sim.save_results(path_save_files, label_simulation, i)
    
    print('The end!')
    
    return LOB_sim.message_df_simulation, LOB_sim.ob_df_simulation, NaiveTrading_class, i_stop_trading, exp_weighted_return_list
