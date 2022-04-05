# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:31:54 2020

@author: willi
"""
import random
import pandas as pd
import numpy as np

def bootSim(tempSample, k, rn):
    random.seed(a=rn)
    simTemps = sorted(random.choices(tempSample, k=k), reverse=True)
    d = {'Temp':[], 'dnF':[]}
    for t in simTemps:
        if t in d['Temp']:
            d['dnF'][d['Temp'].index(t)] += 1
        else:
            if t != np.inf:
                d['Temp'].append(t)
                d['dnF'].append(1)
    d = pd.DataFrame(d)
    d.insert(2, 'nF', d['dnF'].cumsum())
    return d


def statCalc():
    return
