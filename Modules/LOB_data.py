# -*- coding: utf-8 -*-
"""

@author: Adele Ravagnani

Module that defines a class to load and clean a LOBSTER dataset.
This is made up of a "message file" and an "order book file" and it is related to a given asset and a given trading day.

Reference to: https://lobsterdata.com, https://bookdown.org/voigtstefan/advanced_empirical_finance_2023/working-with-lobster.html#read-in-and-process-lobster-files.

"""

import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

#%%
class LOB_data:
    def __init__(self, path_folder, label_message_file, label_ob_file, tick_size):
        """
        Class that allows to load and clean a LOBSTER dataset.

        Parameters
        ----------
        path_folder : str
            Path of the folder where there are the asset's "message file" and "order book file".
        label_message_file : str
            Label of the asset's "message file".
        label_ob_file : str
            Label of the asset's "order book file".
        tick_size : float
            Tick-size of the asset.

        Returns
        -------
        None.

        """
        super().__init__()
        self.path_folder = path_folder
        self.label_message_file = label_message_file
        self.label_ob_file = label_ob_file
        self.message_file = None
        self.ob_file = None
        self.n_levels = None
        self.tick_size = tick_size
        
    def build_class_if_files_already_loaded(self, message_file, ob_file):
        """
        Function which allows to build the class "LOB_data" if the "message file" and the "order book file" were previously obtained.

        Parameters
        ----------
        message_file : pd.DataFrame
            Message file dataframe for the asset considered.
        ob_file : pd.DataFrame
            Order book file dataframe for the asset considered.

        Returns
        -------
        None.

        """
        self.message_file = message_file
        self.ob_file = ob_file
        self.n_levels = len(self.ob_file.columns)//4
        
        return 
    
    def load_LOB_data(self, verbose = True, message_file_with_extra_column = False):
        """
        Function which loads the LOB data.
        We recall that the k-th row in the "message file" describes the LO event causing the change in the LOB from line k-1 to line k in the "orderbook file".
        
        Parameters
        ----------
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.
        message_file_with_extra_column : bool, optional
            If this variable is set to True, it means that the original "message file" contains an extra column and so, it will be dropped. The default is False.
        
        Returns
        -------
        None.

        """
        """
        The first step is to load the "message file".
        
        Message file format
        - 1st entry: time = seconds after midnight
        - 2nd entry: event type (1: submission of LO, 2: partial deletion of LO, 
        3: tot deletion of LO, 4: execution of a visible LO, 
        5: execution of a hidden LO, 6: auction trade, 
        7: trading halt indicator)
        - 3rd entry: order ID
        - 4th entry: size (number of shares)
        - 5th entry: dollar price times 10000
        - 6th entry: direction (-1 sell LO, +1 buy LO)
        """
        
        self.message_file = pd.read_csv(self.path_folder + self.label_message_file, 
                                        header = None)
        
        if message_file_with_extra_column is True:
            self.message_file = self.message_file.iloc[:, :-1]
            
        self.message_file.columns = ['Time', 'Type', 'ID', 
                                'Size', 'Price', 'Direction']
        #dollar price is  multiplied by 10000
        self.message_file['Price'] = self.message_file['Price']/10000
        
        """
        Then, we load the orderbook file.
        
        Order book file format
        ask price level 1, ask size level 1, bid price level 1, bid size level 1, ...
        """
        
        self.ob_file = pd.read_csv(self.path_folder + self.label_ob_file, 
                                   header = None)
        self.n_levels = len(self.ob_file.columns)//4
        
        header_ob_file = []
        for k in range(self.n_levels):
            header_ob_file.append('AskPrice_' + str(k + 1))
            header_ob_file.append('AskSize_' + str(k + 1))
            header_ob_file.append('BidPrice_' + str(k + 1))
            header_ob_file.append('BidSize_' + str(k + 1))
        self.ob_file.columns = header_ob_file
        
        #dollar price is  multiplied by 10000
        for k in range(self.n_levels):
            self.ob_file['AskPrice_' + str(k + 1)] = self.ob_file['AskPrice_' + str(k + 1)]/10000
            self.ob_file['BidPrice_' + str(k + 1)] = self.ob_file['BidPrice_' + str(k + 1)]/10000
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            
            print('Data set lenght:', len(self.ob_file))
            
            print('\nMessage file first lines')
            print(self.message_file.head())
            
            print('\nOrder book file first lines')
            print(self.ob_file.head())
        
        return
    
    def clean_trading_halts(self, verbose = True):
        """
        Function which leans the data from trading halts.
        When trading halts, a message of type '7' is written into the "message file". 
        The corresponding price and trade direction are set to '-1' and all other properties are set to '0'. 
        Should the resume of quoting be indicated by an additional message in NASDAQ's Historical TotalView-ITCH files, another message of type '7' with price '0' is added to the 'message' file. 
        Again, the trade direction is set to '-1' and all other fields are set to '0'. 
        When trading resumes a message of type '7' and price '1' (Trade direction '-1' and all other entries '0') is written to the "message file". 
        For messages of type '7', the corresponding order book rows contain a duplication of the preceding order book state. 
        
        Parameters
        ----------
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """
        
        #check for trading halts
        if 7 in self.message_file['Type'].drop_duplicates().tolist():
            print('Trading halt found in the message file!')
            
            time_start_trading_halt = self.message_file[(self.message_file['Type'] == 7) & (self.message_file['Direction'] == -1) & (self.message_file['Price'] == -1)]['Time'].values[0]
            time_end_trading_halt = self.message_file[(self.message_file['Type'] == 7) & (self.message_file['Direction'] == -1) & (self.message_file['Price'] == +1)]['Time'].values[0]
            ind_to_drop = self.message_file[(self.message_file['Time'] <= time_end_trading_halt) & (self.message_file['Time'] >= time_start_trading_halt)].index
            self.message_file.drop(ind_to_drop, inplace = True)
            self.ob_file.drop(ind_to_drop, inplace = True)
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            
            print('Data set lenght:', len(self.ob_file))
        
        return
    
    def clean_opening_closing_auctions(self, verbose = True):
        """
        Function which removes opening and closing auctions fron the data.
        Sometimes, messages related to the daily opening and closing auction are included in the LOBSTER file 
        (especially during days when trading stops earlier, e.g., before holidays).
        Remove such observations (opening auctions are identified as messages with type == 6 and ID == -1, 
        and closing auctions can be identified as messages with type == 6 and ID == -2)

        Parameters
        ----------
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """
        
        ind_opening_auction = self.message_file[(self.message_file['Type'] == 6) & (self.message_file['ID'] == -1)].index
        ind_closing_auction = self.message_file[(self.message_file['Type'] == 6) & (self.message_file['ID'] == -2)].index
        if len(ind_opening_auction) > 0:
            self.ob_file.drop(ind_opening_auction, inplace = True)
            self.message_file.drop(ind_opening_auction, inplace = True)
        if len(ind_closing_auction) > 0:
            self.ob_file.drop(ind_closing_auction, inplace = True)
            self.message_file.drop(ind_closing_auction, inplace = True)
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            
            print('Data set lenght:', len(self.ob_file))
            
        return
    
    def clean_crossed_prices_obs(self, verbose = True):
        """
        Function which drops observations for which best ask prices are greater than their corresponding best bid prices.
        
        Parameters
        ----------
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """
        
        ind_to_drop = self.ob_file[self.ob_file['AskPrice_1'] < self.ob_file['BidPrice_1']].index
        if len(ind_to_drop) > 0:
            self.ob_file.drop(ind_to_drop, inplace = True)
            self.message_file.drop(ind_to_drop, inplace = True)
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            
            print('Data set lenght:', len(self.ob_file))
        
        return
    
    def handle_splitted_lo_executions(self, verbose = True):
        """
        Function which handles splitted executions of limit orders.
        
        Difference between trades and executions:
       
        The LOBSTER output records limit order executions
        and not what one might intuitively consider trades.
       
        Imagine a volume of 1000 is posted at the best ask
        price. Further, an incoming market buy order of
        volume 1000 is executed against the quote.
       
        The LOBSTER output of this trade depends on the
        composition of the volume at the best ask price.
        Take the following two scenarios with the best ask
         	 volume consisting of ...
       	(a) 1 sell limit order with volume 1000
       	(b) 5 sell limit orders with volume 200 each
         	(ordered according to time of submission)
       
        The LOBSTER output for case ...
          (a) shows one execution of volume 1000. If the
              incoming market order is matched with one
              standing limit order, execution and trade
              coincide.
          (b) shows 5 executions of volume 200 each with the
              same time stamp. The incoming order is matched
              with 5 standing limit orders and triggers 5
              executions.
       
          Bottom line:
          LOBSTER records the exact limit orders against
          which incoming market orders are executed. What
          might be called 'economic' trade size has to be
          inferred from the executions.
        --> from site of LOBSTER
        --> see also https://bookdown.org/voigtstefan/advanced_empirical_finance_2023/working-with-lobster.html#read-in-and-process-lobster-files
       

        Parameters
        ----------
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """
        events_times = self.message_file['Time'].drop_duplicates()
        
        if verbose == True:
            print('Out of %d events, %d are associated to unique times'%(len(self.message_file),
                                                                        len(events_times)))
        
        message_file_grouped_by_time = self.message_file.groupby('Time')
        
        indices_to_drop = []
        for time, group in tqdm(message_file_grouped_by_time):
            if group.shape[0] > 1 and group['Type'].nunique() == 1:
                if group['Type'].unique() == 4 and group['Direction'].nunique() == 1:
                   new_size = group['Size'].sum()
                   new_price = (group['Price']*group['Size']).sum()/(group['Size'].sum())
                   
                   #substitute last event price and size with the weighted average price and the total volume respectively
                   ind_time = np.where(self.message_file['Time'] == time)[0]
                   self.message_file.iat[ind_time[-1], 3] = new_size
                   self.message_file.iat[ind_time[-1], 4] = new_price
                   
                   #indices to drop (we want to drop other events and states)
                   indices_to_drop.append(self.message_file[(self.message_file['Time'] == time)].index[:-1].tolist())
        
        indices_to_drop = [ind for index in indices_to_drop for ind in index]
        #drop other events and states
        self.message_file.drop(indices_to_drop, inplace = True)
        self.ob_file.drop(indices_to_drop, inplace = True)
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            print('New shape is', len(self.ob_file))
            
        return 
    
    def handle_hidden_orders(self, flag_drop = True, verbose = True):
        """
        Functions which handles the hidden orders. They are dropped or handled by reassigning their directions.
        Indeed, the direction of a trade is hard to evaluate (not observable) if the transaction has been executed against a hidden order (`order_type == 5’).
        We create a proxy for the direction based on the executed price relative to the last observed order book snapshot.
        if self.message_file[k] is of type 5, then self.ob_file[k] = self.ob_file[k-1]

        Parameters
        ----------
        flag_drop : bool, optional
            If it is True, the hidden orders are dropped otherwise directions are reassigned. The default is True.
         verbose : bool, optional
             If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """

        hidden_orders = self.message_file[self.message_file['Type'] == 5]
        hidden_orders_ind_pos = np.where(self.message_file['Type'] == 5)[0] #position index vs hidden_orders.index
        
        if flag_drop == True:
            print('Dropping hidden orders ...')
            self.message_file.drop(hidden_orders.index, inplace = True)
            self.ob_file.drop(hidden_orders.index, inplace = True)
        else:    
            print('Reassigning directions to hidden orders ...')
            for k in tqdm(range(len(hidden_orders))):
                order = hidden_orders.iloc[k]
                order_ind_pos = hidden_orders_ind_pos[k]
                order_price = order['Price']
                
                ob_state_before_order = self.ob_file.iloc[order_ind_pos - 1]
                best_ask_before_order = ob_state_before_order['AskPrice_1']
                best_bid_before_order = ob_state_before_order['BidPrice_1']
                mid_price_before_order = (best_bid_before_order + best_ask_before_order)/2
                
                if order_price <= mid_price_before_order:
                    self.message_file.iat[order_ind_pos, -1] = +1
                else:
                    self.message_file.iat[order_ind_pos, -1] = -1
            
            print('Changing labels to hidden orders type such that they are of type 4 as MOs ...')
            self.message_file['Type'].replace(5, 4, inplace = True)
            
            if verbose == True:
                print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
                print('New shape is', len(self.ob_file))
                
        return 
        
             
    def clean_LOB_data(self, flag_drop_hidden_orders = True, verbose = True):
        """
        Function to be executed after loading the dataset i.e. calling the function load_LOB_data.

        Parameters
        ----------
        flag_drop : bool, optional
            If it is True, the hidden orders are dropped otherwise directions are reassigned. The default is True.
         verbose : bool, optional
             If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """

        print('\nCleaning from trading halts ...')
        self.clean_trading_halts(verbose)
        print('\nCleaning from auctions ...')
        self.clean_opening_closing_auctions(verbose)
        print('\nCleaning from crossed price observations ...')
        self.clean_crossed_prices_obs(verbose)
        print('\nHandling splitted LO executions ...')
        self.handle_splitted_lo_executions(verbose)
        print('\nHandling hidden orders ...')
        self.handle_hidden_orders(flag_drop_hidden_orders, verbose)
        

    def load_and_clean_LOB_data(self, flag_drop_hidden_orders = True, verbose = True, message_file_with_extra_column = False):
        """
        Function which loads and cleans the data set by calling several functions of the class.

        Parameters
        ----------
        flag_drop : bool, optional
            If it is True, the hidden orders are dropped otherwise directions are reassigned. The default is True.
         verbose : bool, optional
             If this variable is set to True, information about the process are printed. The default is True.
        message_file_with_extra_column : bool, optional
            If this variable is set to True, it means that the original "message file" contains an extra column and so, it will be dropped. The default is False.
        
        Returns
        -------
        None.

        """
        print('\nLoading message and order book file ...')
        self.load_LOB_data(verbose, message_file_with_extra_column)
        print('\nCleaning message and order book file ...')
        self.clean_LOB_data(flag_drop_hidden_orders, verbose)
        print('\nLoading and cleaning of the dataset completed!')
        
        return
             
    
    def cut_before_and_after_LOB_data(self, minutes_to_cut_beginning, minutes_to_cut_end, verbose = True):
        """
        Function which drops the observations in the first "minutes_to_cut_beginning" and in the last"minutes_to_cut_end".
        
        Parameters
        ----------
        minutes_to_cut_beginning : int
            Minutes at the beginning of the dataset that we want to drop.
        minutes_to_cut_end : int
            Minutes at the end of the dataset that we want to drop.
        verbose : bool, optional
            If this variable is set to True, information about the process are printed. The default is True.

        Returns
        -------
        None.

        """
        
        time_start = self.message_file['Time'].iloc[0]
        time_end = self.message_file['Time'].iloc[-1]
        if verbose == True:
            print('First time available corresponds to%6.2f hours after midnight'%(time_start/3600))
            print('Last time available corresponds to%6.2f hours after midnight'%(time_end/3600))
    
        time_start_new = time_start + 60*minutes_to_cut_beginning
        time_end_new = time_end - 60*minutes_to_cut_end
        
        #----> self.ob_file.iloc[j] is the state of the book after the event self.message_file.iloc[j]
        ind_cut_first_last_hours = (self.message_file['Time'] >= time_start_new) & (self.message_file['Time'] <= time_end_new)
        self.message_file = self.message_file[ind_cut_first_last_hours]
        self.ob_file = self.ob_file[ind_cut_first_last_hours]
        
        if verbose == True:
            print('Now, first time available corresponds to%6.2f hours after midnight'%(time_start_new/3600))
            print('Now, last time available corresponds to%6.2f hours after midnight'%(time_end_new/3600))
        
        if verbose == True:
            print('Check shapes of the new files:', len(self.ob_file) == len(self.message_file))
            print('New shape is', len(self.ob_file))
            
        return
    
    def save_cleaned_and_cut_files(self, path_save):
        """
        Function that allows to save the "message file" and the "order book file" after the cleaning has been performed.

        Parameters
        ----------
        path_save : str
            Path where we want to save the new files of the dataset.

        Returns
        -------
        None.

        """
        self.message_file.to_csv(path_save + 'cleaned_' + self.label_message_file)
        self.ob_file.to_csv(path_save + 'cleaned_' + self.label_ob_file)
        
        return
    
    def obtain_times_datetime_format(self, day):
        """
        Function that allows to add to the "message file" a column with the times converted in the datetime format eg. datetime.datetime(2015, 1, 5, 0, 0, 0) = January 5, 2015 at 0:00.

        Parameters
        ----------
        day : datetime.datetime
            Day of the dataset.

        Returns
        -------
        None.

        """
        
        times = self.message_file['Time']
        shift = day.timestamp()
        times_ = times + shift
        
        times_converted = []
        for i in tqdm(range(len(times))):
            times_converted.append(datetime.datetime.fromtimestamp(times_.iloc[i]))
        
        self.message_file['TimeDatetime']  = times_converted
        
        return
    
    def obtain_ob_time_step(self, time_step):
        """
        Function to obtain the "order book file" not in event time but with a given time step e.g. 60 seconds.
        It needs to be applied after the functon "obtain_times_datetime_format". 
        
        Parameters
        ----------
        time_step : int
            Time step in seconds to sample the order book states.

        Returns
        -------
            
        ob_file_time_step : pd.DataFrame
            Analogous to self.ob_file but sampled in steps equal of time_step and with indices corresponding to the times in datetime format.

        """
        
        if 'TimeDatetime' in self.message_file.columns:
            time_step_datetime = datetime.timedelta(seconds = time_step)
            
            ob_file_copy = self.ob_file.copy()
            ob_file_copy.index = self.message_file['TimeDatetime']
            ob_file_time_step = ob_file_copy.resample(time_step_datetime).apply(lambda x: x.iloc[-1] if len(x) > 0 else [])
        else:
            raise Exception('Before applying this function, the "message file" must have a column with the times in datetime format. So, please appply the function "obtain_times_datetime_format" and then, this function.')
        
        ob_file_time_step = pd.DataFrame(ob_file_time_step.values, 
                            columns = self.ob_file.columns)
        
        return ob_file_time_step.apply(pd.to_numeric)
        
