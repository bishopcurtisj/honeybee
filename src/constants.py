from enum import Enum


## Temp until prezo is imported
class OptionModels(Enum):
    HMC = 1
    LSM = 2
    BSM = 3
    Binomial = 4

class Forecasters(Enum):
    LSTM = 1
    GRU = 2
    RNN = 3
    ARIMA = 4
    ETS = 5
    Naive = 6
    GBM = 7