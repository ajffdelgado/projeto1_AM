# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:54:01 2018

@author: Delgado
"""

import scikit_posthocs as sp
import pandas as pd
import numpy as np
x = np.array([[79.52,92.06,24],[43.38,54.54,31],[79.43,88.60,46]])
sol =  sp.posthoc_nemenyi_friedman(x)