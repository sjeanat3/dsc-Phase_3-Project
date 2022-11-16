import os
from glob import iglob
from typing import BinaryIO, List, Union, Callable
from tabulate import tabulate
from joblib import load, dump

def model_query(model_file:Union[BinaryIO, str], func:Callable, model_subset:List, **kwagrs):
    model = load(model_file)
    
