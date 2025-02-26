# -*- coding: utf-8 -*-
"""
@author: Adele Ravagnani

This is a script which shows how to run a simulation with the Non-Markovian Zero Intelligence model while executing a metaorder.
It produces 2 plots:
1) the mid-price paths for all simulations and its mean (before, during and after the execution);
2) the path of the exponential weighted mid-price return $\bar{R}_t$ (during and after the execution).

"""

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools

import sys
sys.path.append('/adeleravagnani/non-markovian-zero-intelligence-lob-model/Modules')
import NMZI

#%%
def simulation_parallel(lam, mu, delta, mean_inter_arrival_times, number_tick_levels, n_priority_ranks, 
                           parameters_trading_strategy, number_levels_to_store, p0, v0, iterations_before_trading, iterations_after_trading, 
                           iterations_to_equilibrium, beta_exp_weighted_return,
                           intensity_exp_weighted_return, seed):
                           
  np.random.seed(seed)
  
  try:                      
    [message_df_simulation, ob_df_simulation, NaiveTrading_class, 
         i_stop_trading, exp_weighted_return_list] = NMZI.simulate_LOB_and_NaiveTrading(lam, mu, 
                                                                      delta, 
                                                                      mean_inter_arrival_times, 
                                                                      number_tick_levels, n_priority_ranks, 
                                                                      parameters_trading_strategy, 
                                                                      flag_trading_event_time = True,
                                                                      number_levels_to_store = number_levels_to_store ,
                                                                      p0 = p0,
                                                                      v0 = v0,
                                                                      iterations_before_trading = iterations_before_trading,
                                                                      iterations_after_trading = iterations_after_trading, 
                                                                      iterations_to_equilibrium = iterations_to_equilibrium,
                                                                      path_save_files = None, label_simulation = None,
                                                                      beta_exp_weighted_return = beta_exp_weighted_return, 
                                                                      intensity_exp_weighted_return = intensity_exp_weighted_return)
            
    mid_price = message_df_simulation['MidPrice']
    
    return [mid_price, exp_weighted_return_list]  

  except:
    return []
    

#%% Parameters estimated for TSLA on 2015-01-05
lam = 0.0131
mu = 0.0441
delta = 0.1174
v_0 = 101.1057
mean_inter_arrival_times = 0.0951

number_tick_levels = 300
number_levels_to_store = 20
n_priority_ranks = 100
p0 = 20877
v0 = int(v_0)

beta2_exp_weighted_return = 1e-3
beta2_str = str(beta2_exp_weighted_return)
intensity_exp_weighted_return = 1e-3
alpha_str = str(intensity_exp_weighted_return)
trading_interval = 20
direction_trades_user = +1 #buy
total_shares = 2_000

iterations_after_trading = 50_000
iterations_to_equilibrium = 20_000
iterations_before_trading = 20_000
N_sim = 200

folder = '/adeleravagnani/non-markovian-zero-intelligence-lob-model/Examples/'

parameters_trading_strategy = [trading_interval, total_shares, direction_trades_user]  
        
pool = mp.Pool(mp.cpu_count())
store = pool.starmap(simulation_parallel, [(lam, mu, delta, mean_inter_arrival_times, number_tick_levels, n_priority_ranks, 
                       parameters_trading_strategy, number_levels_to_store, p0, v0, iterations_before_trading, iterations_after_trading, 
                       iterations_to_equilibrium, beta2_exp_weighted_return/(trading_interval + 1), 
                       intensity_exp_weighted_return, i) for i in range(N_sim)])
pool.close()

store = [result for result in store if result]

mid_price_n = [store[k][0] for k in range(len(store))]
exp_weighted_return_list_n = [store[k][1] for k in range(len(store))]

mid_price_store = np.mean(mid_price_n, axis = 0)
mid_price_store_std_err = np.std(mid_price_n, axis = 0)/np.sqrt(len(mid_price_n))

exp_weighted_return_list_store = np.mean(exp_weighted_return_list_n, axis = 0)
exp_weighted_return_list_store_std_err = np.std(exp_weighted_return_list_n, axis = 0)/np.sqrt(len(exp_weighted_return_list_n))

plt.figure()
for k in range(len(mid_price_n)):
  plt.plot([k for k in range(1, len(mid_price_n[k]) + 1)], mid_price_n[k], 'k-', linewidth = 0.2)

plt.plot([k for k in range(1, len(mid_price_store) + 1)], mid_price_store, 'r-', linewidth = 0.4)
plt.ylabel('mid-price')
plt.xlabel('market event')
plt.title(r'$\Delta = $' + str(trading_interval) + r', $\beta = $' + beta2_str + r'/($\Delta + 1)$, $\alpha = $' + alpha_str)
plt.savefig(folder + 'mid_price_all_sim.jpeg', dpi = 300)
plt.close()

plt.figure()
plt.grid()
plt.errorbar([k + iterations_before_trading + 1 for k in range(len(exp_weighted_return_list_store))], exp_weighted_return_list_store, yerr = exp_weighted_return_list_store_std_err, fmt = 'o--', elinewidth = 0.6, ms = 2)
plt.ylabel(r'$\bar{R}_t$')
plt.xlabel(r'$t$ (market event)')
plt.title(r'$\Delta = $' + str(trading_interval) + r', $\beta = $' + beta2_str + r'/($\Delta + 1)$, $\alpha = $' + alpha_str)
plt.savefig(folder + 'exp_weighted_return.jpeg', dpi = 300)
plt.close()
