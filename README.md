# non-markovian-santa-fe-lob-simulator

This repository contains the implementation (in Python) of the Non-Markovian Santa Fe (NMSF) model, introduced in the paper: ...

The folder "Modules" consists of 4 modules:
  1) "LOB_data.py" defines a class to load and clean a LOBSTER dataset;
  2) "NMSF_parameters.py" allows to estimate the parameters of the model;
  3) "NMSF.py" is the core module and contains the implementation of the model. Simulations can be performed both without any execution of meta orders and while interacting with the simulator and executing a meta order;
  4) "NMSF_trading.py" contains a class which allows to perform the execution of a meta order with a naive trading strategy and it is employed by "NMSF.py".

The folder "Examples" consists of:
  1) One Jupyter notebook where a LOBSTER data set is loaded, cleaned and the parameters of the model are estimated;
  2) One script which shows how to run a simulation with the Non-Markovian Santa Fe model while executing a meta order.

 If you use the code in your work, please cite its location on Github: https://github.com/adeleravagnani/non-markovian-santa-fe-lob-simulator and the paper: ...

 If you have any inquiries or remarks, do not hesitate to contact the author Adele Ravagnani (adele.ravagnani@sns.it).
