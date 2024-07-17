from dataclasses import dataclass
import numpy as np
#import warnings
from typing import Callable, Protocol, Union, NamedTuple
from .constants import *

class Forecaster(Protocol):

    def __call__(self, *args, **kwargs) -> np.ndarray:
        pass

@dataclass(frozen=True, kw_only=True, slots=True)
class NaiveForecaster:

    def __call__(self, spot: Union[float, np.ndarray], horizon: int) -> np.ndarray:
        return [spot]*horizon
    

REGISTRY = {

    Forecasters.Naive: NaiveForecaster,
}

def create_forecaster(name: Forecasters, **kwargs) -> Forecaster:
    if name not in REGISTRY:
        raise ValueError(f'Forecaster {name} does not exist in the registry')
    return REGISTRY[name](**kwargs)

