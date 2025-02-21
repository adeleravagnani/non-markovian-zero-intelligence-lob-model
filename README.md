# non-markovian-zero-intelligence-lob-model

This repository contains the implementation (in Python) of the Non-Markovian Zero Intelligence (NMZI) model, introduced in the paper: Ravagnani, A. and Lillo, F. (2025). Modeling metaorder impact with a Non-Markovian Zero Intelligence model. link. 

The folder "Modules" consists of 4 modules:
  1) "LOB_data.py" defines a class to load and clean a LOBSTER data set;
  2) "NMZI_parameters.py" allows to estimate the parameters of the model;
  3) "NMZI.py" is the core module and contains the implementation of the model. Simulations can be performed both without any execution of metaorders and while interacting with the model and executing a metaorder;
  4) "NMZI_trading.py" contains a class which allows to perform the execution of a metaorder with a naive trading strategy (costant speed) and it is employed by "NMZI.py".

The folder "Examples" consists of:
  1) One Jupyter notebook where a LOBSTER data set is loaded, cleaned and the parameters of the model are estimated;
  2) One script which shows how to run a simulation with the NMZI model while executing a metaorder.

If you use this code in your work, please cite its location on Github: https://github.com/adeleravagnani/non-markovian-zero-intelligence-lob-model and the paper: Ravagnani, A. and Lillo, F. (2025). Modeling metaorder impact with a Non-Markovian Zero Intelligence model. link. 

 If you have any inquiries or remarks, do not hesitate to contact the author Adele Ravagnani (adele.ravagnani@sns.it).
